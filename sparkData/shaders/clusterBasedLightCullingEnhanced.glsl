#type compute
#version 450
#include "Camera.hglsl"
layout(local_size_x = 1, local_size_y = 16, local_size_z = 1) in;

layout (std140, binding = 0) uniform Camera
{
    CameraData camera;
};

struct ClusterBasedLightCullingData
{
    vec2 pxTileSize;
    uint clusterCountX;
    uint clusterCountY;
    uint clusterCountZ;
    float equation3Part1;
    float equation3Part2;
    uint maxLightCount;
};

layout (std140, binding = 1) uniform AlgorithmData
{
    ClusterBasedLightCullingData algorithmData;
};

struct AABB 
{
    vec3 center;
    float placeholder1;
    vec3 halfSize;
    uint occupancyMask;
};

layout(std430, binding = 0) buffer ClusterData
{
    AABB clusters[];
};

layout(std430, binding = 1) buffer ActiveClusterIndices
{
    uint activeClusterIndices[];
};

struct GlobalIndicesOffset
{
    uint globalPointLightIndicesOffset;
    uint globalSpotLightIndicesOffset;
    uint globalLightProbeIndicesOffset;
};

layout(std430, binding = 2) buffer GlobalPointLightIndices
{
    uint globalPointLightIndices[];
};

layout(std430, binding = 3) buffer GlobalSpotLightIndices
{
    uint globalSpotLightIndices[];
};

layout(std430, binding = 4) buffer GlobalLightProbeIndices
{
    uint globalLightProbeIndices[];
};

struct LightIndicesBufferMetadata
{
    uint pointLightIndicesOffset;
    uint pointLightCount;
    uint spotLightIndicesOffset;
    uint spotLightCount;
    uint lightProbeIndicesOffset;
    uint lightProbeCount;
};

layout(std430, binding = 5) buffer PerClusterGlobalLightIndicesBufferMetadata
{
    LightIndicesBufferMetadata lightIndicesBufferMetadata[];
};

struct PointLight {
    vec4 positionAndRadius; // radius in w component
    vec3 color;
    float nothing2;
    mat4 modelMat;
};

struct SpotLight {
    vec3 position;
    float cutOff;
    vec3 color;
    float outerCutOff;
    vec3 direction;
    float maxDistance;
    vec4 boundingSphere; //xyz - sphere center, w - radius 
};

struct LightProbe {
    uvec2 irradianceCubemapHandle; //int64_t handle
    uvec2 prefilterCubemapHandle; //int64_t handle
    vec4 positionAndRadius;
    float fadeDistance;
    float padding1;
    float padding2;
    float padding3;
};

layout(std430, binding = 1) readonly buffer PointLightData
{
    PointLight pointLights[];
};

layout(std430, binding = 1) readonly buffer SpotLightData
{
    SpotLight spotLights[];
};

layout(std430, binding = 1) readonly buffer LightProbeData
{
    LightProbe lightProbes[];
};

#define MAX_LIGHTS 255

shared uint pointLightCount;
shared uint spotLightCount;
shared uint lightProbeCount;
shared uint clusterIndex;

shared uvec3 minBitIndices;
shared uvec3 maxBitIndices;

shared AABB cluster;

bool testSphereVsAABB(const vec3 sphereCenter, const float sphereRadius, const vec3 AABBCenter, const vec3 AABBHalfSize)
{
    vec3 delta = max(vec3(0), abs(AABBCenter - sphereCenter) - AABBHalfSize);
    float distSq = dot(delta, delta);
    return distSq <= (sphereRadius * sphereRadius);
}

bool spotLightConeVsAABB(const SpotLight spotLight, const vec3 aabbCenter, const float aabbSphereRadius)
{
    const vec3 position = (camera.view * vec4(spotLight.boundingSphere.xyz, 1.0f)).xyz;
    const vec3 direction = normalize((camera.view * vec4(spotLight.direction.xyz, 0.0f)).xyz);
    const float radius = spotLight.boundingSphere.w;
    const float angle = acos(spotLight.outerCutOff);
    const float range = spotLight.maxDistance;

    const vec3 v = aabbCenter - position;
    const float lenSq = dot(v, v);
    const float v1Len = dot(v, direction);
    const float distanceClosestPoint = cos(angle) * sqrt(lenSq - v1Len * v1Len) - v1Len * sin(angle);
    const bool angleCull = distanceClosestPoint > aabbSphereRadius;
    const bool frontCull = v1Len > aabbSphereRadius + range;
    const bool backCull = v1Len < -aabbSphereRadius;
    return !(angleCull || frontCull || backCull);
}

void cullPointLights()
{
    const uint offset = clusterIndex * algorithmData.maxLightCount;

    //const vec3 clusterRangeReciprocal = 10.0f / (cluster.halfSize * 2.0f);

    for (uint i = gl_LocalInvocationIndex; i < pointLights.length(); i += gl_WorkGroupSize.y)
    {
        PointLight p = pointLights[i];
        const vec3 pPos = (camera.view * vec4(p.positionAndRadius.xyz, 1.0f)).xyz;
        const float pRadius = p.positionAndRadius.w;

        uint lightCount = 0;
        if (testSphereVsAABB(pPos, pRadius, cluster.center, cluster.halfSize))
        {
            lightCount = min(atomicAdd(pointLightCount, 1), algorithmData.maxLightCount);
            globalPointLightIndices[offset + lightCount] = i;
        }

        if(lightCount == MAX_LIGHTS)
            break;
    }

    barrier();

    if (gl_LocalInvocationIndex == 0)
    {
        lightIndicesBufferMetadata[clusterIndex].pointLightIndicesOffset = offset;
        lightIndicesBufferMetadata[clusterIndex].pointLightCount = pointLightCount;
    }
}

void cullSpotLights()
{
    const uint offset = clusterIndex * algorithmData.maxLightCount;

    const float aabbSphereRadius = length(cluster.halfSize);
    for (uint i = gl_LocalInvocationIndex; i < spotLights.length(); i += gl_WorkGroupSize.y)
    {
        const SpotLight s = spotLights[i];
        uint lightCount = 0;
        if(spotLightConeVsAABB(s, cluster.center, aabbSphereRadius))
        {
            lightCount = min(atomicAdd(spotLightCount, 1), algorithmData.maxLightCount);
            globalSpotLightIndices[offset + lightCount] = i;
        }

        if(lightCount == MAX_LIGHTS)
            break;
    }

    barrier();

    if (gl_LocalInvocationIndex == 0)
    {
        lightIndicesBufferMetadata[clusterIndex].spotLightIndicesOffset = offset;
        lightIndicesBufferMetadata[clusterIndex].spotLightCount = spotLightCount;
    }
}

void cullLightProbes()
{
    const uint offset = clusterIndex * algorithmData.maxLightCount;

    for (uint i = gl_LocalInvocationIndex; i < lightProbes.length(); i += gl_WorkGroupSize.y)
    {
        LightProbe l = lightProbes[i];
        const vec3 lPos = (camera.view * vec4(l.positionAndRadius.xyz, 1.0f)).xyz;
        const float lRadius = l.positionAndRadius.w;
        uint lightCount = 0;
        if (testSphereVsAABB(lPos, lRadius, cluster.center, cluster.halfSize))
        {
            lightCount = min(atomicAdd(lightProbeCount, 1), algorithmData.maxLightCount);
            globalLightProbeIndices[offset + lightCount] = i;
        }

        if(lightCount == MAX_LIGHTS)
            break;
    }

    barrier();

    if (gl_LocalInvocationIndex == 0)
    {
        lightIndicesBufferMetadata[clusterIndex].lightProbeIndicesOffset = offset;
        lightIndicesBufferMetadata[clusterIndex].lightProbeCount = lightProbeCount;
    }
}

void main()
{
    if (gl_LocalInvocationIndex == 0)
    {
        clusterIndex = activeClusterIndices[gl_GlobalInvocationID.x];
        pointLightCount = 0;
        spotLightCount = 0;
        lightProbeCount = 0;

        minBitIndices = uvec3(9);
        maxBitIndices = uvec3(0);
        cluster = clusters[clusterIndex];
    }

    barrier();

    if (gl_LocalInvocationIndex < 10)
    {
        const uint bitmap = cluster.occupancyMask;

        uvec3 clusterOccupancy = uvec3(0);
        clusterOccupancy.x = bitfieldExtract(bitmap, 0, 10) & 0x3FF;
        clusterOccupancy.y = bitfieldExtract(bitmap, 10, 10) & 0x3FF;
        clusterOccupancy.z = bitfieldExtract(bitmap, 20, 10) & 0x3FF;

        if (bool((clusterOccupancy.x >> gl_LocalInvocationIndex) & 0x1) )
        {
            atomicMin(minBitIndices.x, gl_LocalInvocationIndex);
            atomicMax(maxBitIndices.x, gl_LocalInvocationIndex);
        }

        if (bool((clusterOccupancy.y >> gl_LocalInvocationIndex) & 0x1) )
        {
            atomicMin(minBitIndices.y, gl_LocalInvocationIndex);
            atomicMax(maxBitIndices.y, gl_LocalInvocationIndex);
        }

        if (bool((clusterOccupancy.z >> gl_LocalInvocationIndex) & 0x1) )
        {
            atomicMin(minBitIndices.z, gl_LocalInvocationIndex);
            atomicMax(maxBitIndices.z, gl_LocalInvocationIndex);
        }
    }

    barrier();

    if (gl_LocalInvocationIndex == 0)
    {
        const vec3 clusterSegmentSize = (cluster.halfSize * 2.0f) * 0.1f;
        const vec3 clusterMin = (cluster.center - cluster.halfSize) + minBitIndices * clusterSegmentSize;
        const vec3 clusterMax = (cluster.center - cluster.halfSize) + (maxBitIndices + 1) * clusterSegmentSize;

        const vec3 clusterHalfSize = (clusterMax - clusterMin) * 0.5f;
        const vec3 clusterCenter = clusterMin + clusterHalfSize;

        cluster.center = clusterCenter;
        cluster.halfSize = clusterHalfSize;
    }

    barrier();

    cullPointLights();
    cullSpotLights();
    cullLightProbes();

    clusters[clusterIndex].occupancyMask = 0;
}