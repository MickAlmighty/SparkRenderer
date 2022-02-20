#type compute
#version 450
#include "Camera.hglsl"
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D depthTexture;

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

layout(std430, binding = 0) buffer ActiveClusters
{
    bool activeClusters[];
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

layout(std430, binding = 1) buffer PerClusterGlobalLightIndicesBufferMetadata
{
    LightIndicesBufferMetadata lightIndicesBufferMetadata[];
};

vec3 fromPxToViewSpace(vec2 pixelCoords, vec2 screenSize, float depth)
{
    pixelCoords.xy /= screenSize;
    vec4 ndcCoords = vec4(pixelCoords.xy * 2.0f - 1.0f, depth, 1.0f);
    ndcCoords = camera.invertedProjection * ndcCoords;
    
    vec3 viewSpacePosition = ndcCoords.xyz / ndcCoords.w;
    return viewSpacePosition;
}

uint getZSlice(float viewSpaceDepth)
{
    return uint(log(abs(viewSpaceDepth)) * algorithmData.equation3Part1 - algorithmData.equation3Part2);
}

uint calculateClusterIndex(uint clusterZ)
{
    const uint clustersX = algorithmData.clusterCountX;
    const uint clustersY = algorithmData.clusterCountY;
    const uint screenSliceOffset = clustersX * clustersY * clusterZ;

    const uvec2 clusterAssignmentXY = uvec2(vec2(gl_GlobalInvocationID.xy) / algorithmData.pxTileSize);
    const uint onScreenSliceIndex = clusterAssignmentXY.y * clustersX + clusterAssignmentXY.x;

    return screenSliceOffset + onScreenSliceIndex;
}

const LightIndicesBufferMetadata clearDataObject = LightIndicesBufferMetadata(0, 0, 0, 0, 0, 0);

void main()
{
    const vec2 texSize = vec2(textureSize(depthTexture, 0));
    const ivec2 texCoords = ivec2(gl_GlobalInvocationID.xy);
    const float depth = texelFetch(depthTexture, texCoords, 0).x;

    if (depth != 0.0f)
    {
        const float viewSpaceDepth = fromPxToViewSpace(texCoords, texSize, depth).z;

        const uint clusterZ = getZSlice(viewSpaceDepth);
        const uint clusterIndex = calculateClusterIndex(clusterZ);

        activeClusters[clusterIndex] = true;
        lightIndicesBufferMetadata[clusterIndex] = clearDataObject;
    }
}