#type compute
#version 450
#include "Camera.hglsl"
#include "ClusterBasedLightCullingData.hglsl"
#include "GlobalLightIndices.hglsl"
#include "PointLight.hglsl"
#include "SpotLight.hglsl"
#include "LightProbe.hglsl"
layout(local_size_x = 1, local_size_y = 16, local_size_z = 1) in;

layout (std140, binding = 0) uniform Camera
{
    CameraData camera;
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

struct LightIndicesBufferMetadata
{
    uint lightIndicesOffset;
    uint pointLightCount;
    uint spotLightCount;
    uint lightProbeCount;
};

layout(std430, binding = 3) buffer PerClusterGlobalLightIndicesBufferMetadata
{
    LightIndicesBufferMetadata lightIndicesBufferMetadata[];
};

layout(std430, binding = 4) readonly buffer PointLightData
{
    PointLight pointLights[];
};

layout(std430, binding = 5) readonly buffer SpotLightData
{
    SpotLight spotLights[];
};

layout(std430, binding = 6) readonly buffer LightProbeData
{
    LightProbe lightProbes[];
};

shared uint pointLightCount;
shared uint spotLightCount;
shared uint lightProbeCount;
shared uint clusterIndex;
shared uint offset;

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
    for (uint i = gl_LocalInvocationIndex; i < pointLights.length(); i += gl_WorkGroupSize.y)
    {
        PointLight p = pointLights[i];
        const vec3 pPos = (camera.view * vec4(p.positionAndRadius.xyz, 1.0f)).xyz;
        const float pRadius = p.positionAndRadius.w;

        uint lightCount = 0;
        if (testSphereVsAABB(pPos, pRadius, cluster.center, cluster.halfSize))
        {
            lightCount = min(atomicAdd(pointLightCount, 1), algorithmData.maxLightCount);
            insertPointLightIndex(offset + lightCount, i);
        }

        if (lightCount == algorithmData.maxLightCount)
            break;
    }
}

void cullSpotLights()
{
    const float aabbSphereRadius = length(cluster.halfSize);
    for (uint i = gl_LocalInvocationIndex; i < spotLights.length(); i += gl_WorkGroupSize.y)
    {
        const SpotLight s = spotLights[i];
        uint lightCount = 0;
        if (spotLightConeVsAABB(s, cluster.center, aabbSphereRadius))
        {
            lightCount = min(atomicAdd(spotLightCount, 1), algorithmData.maxLightCount);
            insertSpotLightIndex(offset + lightCount, i);
        }

        if (lightCount == algorithmData.maxLightCount)
            break;
    }
}

void cullLightProbes()
{
    for (uint i = gl_LocalInvocationIndex; i < lightProbes.length(); i += gl_WorkGroupSize.y)
    {
        LightProbe l = lightProbes[i];
        const vec3 lPos = (camera.view * vec4(l.positionAndRadius.xyz, 1.0f)).xyz;
        const float lRadius = l.positionAndRadius.w;
        uint lightCount = 0;
        if (testSphereVsAABB(lPos, lRadius, cluster.center, cluster.halfSize))
        {
            lightCount = min(atomicAdd(lightProbeCount, 1), algorithmData.maxNumberOfLightProbes);
            insertLightProbeIndex(offset + lightCount, i);
        }

        if (lightCount == algorithmData.maxNumberOfLightProbes)
            break;
    }
}

void getClusterMinMaxDimsBasedOnOccupancyMask()
{
    const uint bitmap = cluster.occupancyMask;

    uvec3 clusterOccupancy = uvec3(0);
    clusterOccupancy.x = bitfieldExtract(bitmap, 0, 10);
    clusterOccupancy.y = bitfieldExtract(bitmap, 10, 10);
    clusterOccupancy.z = bitfieldExtract(bitmap, 20, 10);

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

void createTightClusterAABB()
{
    const vec3 clusterSegmentSize = (cluster.halfSize * 2.0f) * 0.1f;
    const vec3 clusterMin = (cluster.center - cluster.halfSize) + minBitIndices * clusterSegmentSize;
    const vec3 clusterMax = (cluster.center - cluster.halfSize) + (maxBitIndices + 1) * clusterSegmentSize;

    const vec3 clusterHalfSize = (clusterMax - clusterMin) * 0.5f;
    const vec3 clusterCenter = clusterMin + clusterHalfSize;

    cluster.center = clusterCenter;
    cluster.halfSize = clusterHalfSize;
}

void main()
{
    if (gl_LocalInvocationIndex == 0)
    {
        clusterIndex = activeClusterIndices[gl_GlobalInvocationID.x];
        offset = clusterIndex * algorithmData.maxLightCount;
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
        getClusterMinMaxDimsBasedOnOccupancyMask();
    }

    barrier();

    if (gl_LocalInvocationIndex == 0)
    {
        createTightClusterAABB();
    }

    barrier();

    cullPointLights();
    cullSpotLights();
    cullLightProbes();

    barrier();

    if (gl_LocalInvocationIndex == 0)
    {
        clusters[clusterIndex].occupancyMask = 0;
        lightIndicesBufferMetadata[clusterIndex].lightIndicesOffset = offset;
        lightIndicesBufferMetadata[clusterIndex].pointLightCount = pointLightCount;
        lightIndicesBufferMetadata[clusterIndex].spotLightCount = spotLightCount;
        lightIndicesBufferMetadata[clusterIndex].lightProbeCount = lightProbeCount;
    }
}