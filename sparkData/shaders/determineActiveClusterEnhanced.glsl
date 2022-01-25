#type compute
#version 450
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D depthTexture;

layout (std140) uniform Camera
{
    vec4 pos;
    mat4 view;
    mat4 projection;
    mat4 invertedView;
    mat4 invertedProjection;
    mat4 viewProjection;
    mat4 invertedViewProjection;
    float nearZ;
    float farZ;
} camera;

layout (std140) uniform AlgorithmData
{
    vec2 pxTileSize;
    uint clusterCountX;
    uint clusterCountY;
    uint clusterCountZ;
    float equation3Part1;
    float equation3Part2;
    uint maxLightCount;
} algorithmData;

struct AABB 
{
    vec4 center;
    vec3 halfSize;
    uint occupancyMask;
};

layout(std430) buffer ClusterData
{
    AABB clusters[];
};

layout(std430) buffer ActiveClusters
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

layout(std430) buffer PerClusterGlobalLightIndicesBufferMetadata
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
        const vec3 viewSpacePosition = fromPxToViewSpace(texCoords, texSize, depth);

        const uint clusterZ = getZSlice(viewSpacePosition.z);
        const uint clusterIndex = calculateClusterIndex(clusterZ);

        {
            const AABB aabb = clusters[clusterIndex];
            const vec3 clusterRangeReciprocal = 10.0f / (aabb.halfSize * 2.0f);
            const vec3 clusterMin = aabb.center.xyz - aabb.halfSize;
            const uvec3 maskCellIndex = uvec3(max(vec3(0), min(vec3(10), floor((viewSpacePosition - clusterMin) * clusterRangeReciprocal))));

            const uvec3 mask = 1 << maskCellIndex;

            uint clusterOccupancyMask = 0;
            clusterOccupancyMask = bitfieldInsert(clusterOccupancyMask, mask.x, 0, 10);
            clusterOccupancyMask = bitfieldInsert(clusterOccupancyMask, mask.y, 10, 10);
            clusterOccupancyMask = bitfieldInsert(clusterOccupancyMask, mask.z, 20, 10);

            atomicOr(clusters[clusterIndex].occupancyMask, clusterOccupancyMask);
        }

        activeClusters[clusterIndex] = true;
        lightIndicesBufferMetadata[clusterIndex] = clearDataObject;
    }
}