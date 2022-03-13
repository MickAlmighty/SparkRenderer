#type compute
#version 450
#include "Camera.hglsl"
#include "ClusterBasedLightCullingData.hglsl"
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

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
    vec4 center;
    vec4 halfSize;
};

layout(std430, binding = 0) buffer ClusterData
{
    AABB clusters[];
};

vec3 fromPxToViewSpace(vec4 pixelCoords, vec2 screenSize)
{
    pixelCoords.xy /= screenSize;
    vec4 ndcCoords = vec4(pixelCoords.xy * 2.0f - 1.0f, pixelCoords.zw);
    ndcCoords = camera.invertedProjection * ndcCoords;
    
    vec3 viewSpacePosition = ndcCoords.xyz / ndcCoords.w;
    return viewSpacePosition;
}

vec3 pointOnPlanePerpendicularToCameraFront(float planeZ, vec3 pointOnLine)
{
    float t = planeZ / pointOnLine.z;
    return pointOnLine * t;
}

AABB createTileAABB(const float nearPlaneZ, const float farPlaneZ)
{
    const vec2 tilesCount = vec2(algorithmData.clusterCountX, algorithmData.clusterCountY);
    const vec2 screenSize = algorithmData.pxTileSize * tilesCount;

    const vec2 bottomLeftPointOnTile = gl_GlobalInvocationID.xy * algorithmData.pxTileSize;
    const vec2 upperRightPointOnTile = (gl_GlobalInvocationID.xy + 1) * algorithmData.pxTileSize;
    const float depthBufferFar = 0.0f;
    const vec3 cameraFarPlaneMinPoint = fromPxToViewSpace(vec4(bottomLeftPointOnTile, depthBufferFar, 1), screenSize);
    const vec3 cameraFarPlaneMaxPoint = fromPxToViewSpace(vec4(upperRightPointOnTile, depthBufferFar, 1), screenSize);

    const vec3 nearPlaneMinPoint = pointOnPlanePerpendicularToCameraFront(nearPlaneZ, cameraFarPlaneMinPoint);
    const vec3 farPlaneMinPoint = pointOnPlanePerpendicularToCameraFront(farPlaneZ, cameraFarPlaneMinPoint);
    const vec3 nearPlaneMaxPoint = pointOnPlanePerpendicularToCameraFront(nearPlaneZ, cameraFarPlaneMaxPoint);
    const vec3 farPlaneMaxPoint = pointOnPlanePerpendicularToCameraFront(farPlaneZ, cameraFarPlaneMaxPoint);

    //AABB's min-max points
    vec3 minPoint = min(min(nearPlaneMinPoint, nearPlaneMaxPoint), min(farPlaneMinPoint, farPlaneMaxPoint));
    vec3 maxPoint = max(max(nearPlaneMinPoint, nearPlaneMaxPoint), max(farPlaneMinPoint, farPlaneMaxPoint));

    vec3 aabbHalfSize = abs(maxPoint - minPoint) * 0.5;
    vec3 aabbCenter = minPoint + aabbHalfSize;

    return AABB(vec4(aabbCenter, 0.0f), vec4(aabbHalfSize, 0.0f));
}

uint calculateIndex()
{
    const uvec2 tilesCount = uvec2(algorithmData.clusterCountX, algorithmData.clusterCountY);
    uint screenSliceOffset = tilesCount.x * tilesCount.y * gl_GlobalInvocationID.z;
    uint onScreenSliceIndex = gl_GlobalInvocationID.y * tilesCount.x + gl_GlobalInvocationID.x;
    return screenSliceOffset + onScreenSliceIndex;
}

void main()
{
    float clusterMinZ = -1.0f * camera.nearZ * pow(camera.farZ / camera.nearZ, float(gl_WorkGroupID.z) / float(gl_NumWorkGroups.z));
    float clusterMaxZ = -1.0f * camera.nearZ * pow(camera.farZ / camera.nearZ, float(gl_WorkGroupID.z + 1) / float(gl_NumWorkGroups.z));

    uint index = calculateIndex();
    clusters[index] = createTileAABB(clusterMinZ, clusterMaxZ);
}