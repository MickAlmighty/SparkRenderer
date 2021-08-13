#type compute
#version 450
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(binding = 0) uniform sampler2D depthTexture;

uniform vec2 tileSize;
//uniform float equation3Part1;
//uniform float equation3Part2;

layout (std140) uniform Camera
{
    vec4 pos;
    mat4 view;
    mat4 projection;
    mat4 invertedView;
    mat4 invertedProjection;
    float nearZ;
    float farZ;
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
    const float clustersZ = 32.0f;
    const float logNearByFar = log(camera.farZ / camera.nearZ);
    const float equation3Part1 = clustersZ / logNearByFar;
    const float equation3Part2 = clustersZ * log(camera.nearZ) / logNearByFar;
    return uint(log(abs(viewSpaceDepth)) * equation3Part1 - equation3Part2);
}

uint calculateClusterIndex(uint clusterZ)
{
    const uint clustersX = 64;
    const uint clustersY = 64;
    uint screenSliceOffset = clustersX * clustersY * clusterZ;

    uvec2 clusterAssignmentXY = uvec2(vec2(gl_GlobalInvocationID.xy) / tileSize);
    uint onScreenSliceIndex = clusterAssignmentXY.x * clustersY + clusterAssignmentXY.y;

    return screenSliceOffset + onScreenSliceIndex;
}

const LightIndicesBufferMetadata clearDataObject = LightIndicesBufferMetadata(0, 0, 0, 0, 0, 0);

void main()
{
    const vec2 texSize = vec2(textureSize(depthTexture, 0));
    const ivec2 texCoords = ivec2(gl_GlobalInvocationID.xy);

    const float depth = texelFetch(depthTexture, texCoords, 0).x;

    if (depth == 0.0f)
        return;

    const float viewSpaceDepth = fromPxToViewSpace(texCoords, texSize, depth).z;

    const uint clusterZ = getZSlice(viewSpaceDepth);
    uint clusterIndex = calculateClusterIndex(clusterZ);

    activeClusters[clusterIndex] = true;

    lightIndicesBufferMetadata[clusterIndex] = clearDataObject;
}