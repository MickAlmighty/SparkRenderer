#type compute
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (std140) uniform Camera
{
    vec4 pos;
    mat4 view;
    mat4 projection;
    mat4 invertedView;
    mat4 invertedProjection;
    mat4 viewProjection;
    mat4 invertedViewProjection;
} camera;

struct AABB 
{
    vec3 center;
    float placeholder1;
    vec3 halfSize;
    float placeholder2;
};

layout(std430) buffer ClusterData
{
    AABB clusters[];
};

layout(std430) buffer ActiveClusterIndices
{
    uint activeClusterIndices[];
};

struct GlobalIndicesOffset
{
    uint globalPointLightIndicesOffset;
    uint globalSpotLightIndicesOffset;
    uint globalLightProbeIndicesOffset;
};

layout(std430) buffer GlobalPointLightIndices
{
    uint globalPointLightIndices[];
};

layout(std430) buffer GlobalSpotLightIndices
{
    uint globalSpotLightIndices[];
};

layout(std430) buffer GlobalLightProbeIndices
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

layout(std430) buffer PerClusterGlobalLightIndicesBufferMetadata
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

layout(std430) readonly buffer PointLightData
{
    PointLight pointLights[];
};

layout(std430) readonly buffer SpotLightData
{
    SpotLight spotLights[];
};

layout(std430) readonly buffer LightProbeData
{
    LightProbe lightProbes[];
};

#define MAX_LIGHTS 255

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

void cullPointLights(uint clusterIndex)
{
    AABB cluster = clusters[clusterIndex];
    const uint offset = clusterIndex * 256;
    uint lightCount = 0;

    for (int i = 0; i < pointLights.length(); ++i)
    {
        PointLight p = pointLights[i];
        const vec3 pPos = (camera.view * vec4(p.positionAndRadius.xyz, 1.0f)).xyz;
        const float pRadius = p.positionAndRadius.w;
        if (testSphereVsAABB(pPos, pRadius, cluster.center, cluster.halfSize))
        {
            globalPointLightIndices[offset + lightCount] = i;
            lightCount += 1;
        }

        if(lightCount == MAX_LIGHTS)
            break;
    }

    lightIndicesBufferMetadata[clusterIndex].pointLightIndicesOffset = offset;
    lightIndicesBufferMetadata[clusterIndex].pointLightCount = lightCount;
}

void cullSpotLights(uint clusterIndex)
{
    AABB cluster = clusters[clusterIndex];
    const uint offset = clusterIndex * 256;
    uint lightCount = 0;

    const float aabbSphereRadius = length(cluster.halfSize);
    for (uint i = 0; i < spotLights.length(); ++i)
    {
        const SpotLight s = spotLights[i];
        if(spotLightConeVsAABB(s, cluster.center, aabbSphereRadius))
        {
            globalSpotLightIndices[offset + lightCount] = i;
            lightCount += 1;
        }

        if(lightCount == MAX_LIGHTS)
            break;
    }

    lightIndicesBufferMetadata[clusterIndex].spotLightIndicesOffset = offset;
    lightIndicesBufferMetadata[clusterIndex].spotLightCount = lightCount;
}

void cullLightProbes(uint clusterIndex)
{
    AABB cluster = clusters[clusterIndex];
    const uint offset = clusterIndex * 256;
    uint lightCount = 0;

    for (int i = 0; i < lightProbes.length(); ++i)
    {
        LightProbe l = lightProbes[i];
        const vec3 lPos = (camera.view * vec4(l.positionAndRadius.xyz, 1.0f)).xyz;
        const float lRadius = l.positionAndRadius.w;
        if (testSphereVsAABB(lPos, lRadius, cluster.center, cluster.halfSize))
        {
            globalLightProbeIndices[offset + lightCount] = i;
            lightCount += 1;
        }

        if(lightCount == MAX_LIGHTS)
            break;
    }

    lightIndicesBufferMetadata[clusterIndex].lightProbeIndicesOffset = offset;
    lightIndicesBufferMetadata[clusterIndex].lightProbeCount = lightCount;
}

void main()
{
    const uint clusterIndex = activeClusterIndices[gl_GlobalInvocationID.x];
    cullPointLights(clusterIndex);
    cullSpotLights(clusterIndex);
    cullLightProbes(clusterIndex);
}