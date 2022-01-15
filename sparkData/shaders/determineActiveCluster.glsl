#type compute
#version 450
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D depthTexture;

uniform vec2 tileSize;

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
    float equation3Part1;
    float equation3Part2;
} camera;

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
    return uint(log(abs(viewSpaceDepth)) * camera.equation3Part1 - camera.equation3Part2);
}

uint calculateClusterIndex(uint clusterZ)
{
    const uint clustersX = 64;
    const uint clustersY = 64;
    uint screenSliceOffset = clustersX * clustersY * clusterZ;

    uvec2 clusterAssignmentXY = uvec2(vec2(gl_GlobalInvocationID.xy) / tileSize);
    uint onScreenSliceIndex = clusterAssignmentXY.y * clustersX + clusterAssignmentXY.x;

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