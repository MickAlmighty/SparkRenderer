#type compute
#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) uniform sampler2D depthTexture; 

#define DEBUG
#ifdef DEBUG
    layout(rgba16f, binding = 5) uniform image2D lightCountImage;
#endif

shared uint minDepth;
shared uint maxDepth;
shared uint tileDepthMask;
shared uint pointLightCount;
shared uint spotLightCount;
shared uint lightProbesCount;
shared uint pixelsToShade;
shared uint numberOfLightsIndex;
shared uint lightBeginIndex;

#define MAX_WORK_GROUP_SIZE 16
#define THREADS_PER_TILE MAX_WORK_GROUP_SIZE * MAX_WORK_GROUP_SIZE
#define M_PI 3.14159265359 

layout (std140) uniform Camera
{
    vec4 pos;
    mat4 view;
    mat4 projection;
    mat4 invertedView;
    mat4 invertedProjection;
} camera;

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
    ivec2 irradianceCubemapHandle; //int64_t handle
    ivec2 prefilterCubemapHandle; //int64_t handle
    vec3 position;
    float radius;
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

layout(std430) buffer PointLightIndices
{
    uint pointLightIndices[];
};

layout(std430) buffer SpotLightIndices
{
    uint spotLightIndices[];
};

layout(std430) buffer LightProbeIndices
{
    uint lightProbeIndices[];
};

struct AABB 
{
    vec3 aabbCenter;
    vec3 aabbHalfSize;
};

struct Frustum
{
    vec4 planes[6];
};

//light culling functions
AABB createTileAABB(float minDepthZ, float maxDepthZ, vec2 texSize);
Frustum createTileFrustum(float minDepthZ, float maxDepthZ, vec2 texSize);
bool testSphereVsAABB(vec3 sphereCenter, float sphereRadius, vec3 AABBCenter, vec3 AABBHalfSize);
vec3 fromNdcToViewSpace(vec4 ndcPoint);
float pixToNDC(uint point, float dimSize);
vec3 createPlane(vec3 p1, vec3 p2);
vec4 createPlanePerpendicularToView(vec3 dir1, vec3 dir2, vec3 pointOnPlane);
float getDistanceFromPlane(vec3 position, vec3 plane);
float getDistanceFromPerpendicularPlane(vec3 position, vec4 perpendicularPlane);
void cullPointLights(AABB tileAABB, float minDepthVS, float depthRangeRecip);
void cullPointLights(Frustum tileFrustum, float minDepthVS, float depthRangeRecip);
void cullSpotLights(AABB tileAABB);
void cullLightProbes(AABB tileAABB, float minDepthVS, float depthRangeRecip);
//

vec3 worldPosFromDepth(float depth, vec2 texCoords);

void main()
{
    vec2 texSize = vec2(textureSize(depthTexture, 0));
    ivec2 texCoords = ivec2(gl_GlobalInvocationID.xy);

    if (gl_LocalInvocationIndex == 0)
    {
        pointLightCount = 0;
        spotLightCount = 0;
        lightProbesCount = 0;
        pixelsToShade = 0;
        minDepth = 0xffffffffu;
        maxDepth = 0;
        tileDepthMask = 0;

        uint nrOfElementsInColum = 256 * gl_NumWorkGroups.y;
        uint startIndex = nrOfElementsInColum * gl_WorkGroupID.x + 256 * gl_WorkGroupID.y;
        numberOfLightsIndex = startIndex;
        lightBeginIndex = startIndex + 1;
#ifdef DEBUG
        imageStore(lightCountImage, ivec2(gl_WorkGroupID.xy), vec4(0));
#endif
    }

    barrier();

    float depthFloat = texelFetch(depthTexture, texCoords, 0).x;

    if (depthFloat != 0.0f)
    {
        atomicAdd(pixelsToShade, 1);
    }

    barrier();

    if (pixelsToShade == 0)
    {
        return;
    }

    uint depthInt = uint(depthFloat * 0xffffffffu);

    //Calcutate the depth range of this tile
    //Depth values are reversed, that means that zFar = 0, zNear = 1
    atomicMin(minDepth, depthInt); //tile zFar
    atomicMax(maxDepth, depthInt); //tile zNear

    // wait for process all invocations
    barrier();

    //convert min and max depth back to the float
    float maxDepthZ = float(float(maxDepth) / float(0xffffffffu));
    float minDepthZ = float(float(minDepth) / float(0xffffffffu));

    float minDepthVS = fromNdcToViewSpace(vec4(0.0f, 0.0f, max(minDepthZ, 0.00001f), 1.0f)).z;
    float maxDepthVS = fromNdcToViewSpace(vec4(0.0f, 0.0f, maxDepthZ, 1.0f)).z;
    float pixelDepthVS = fromNdcToViewSpace(vec4(0.0f, 0.0f, depthFloat, 1.0f)).z;

    //assigning linear depth [0,32] for each pixel from tile view space depth range
    float depthRangeRecip = 32.0f / (maxDepthVS - minDepthVS); // positive result
    //calculating proper bit index  
    uint depthMaskCellIndex = uint(max(0, min(32, floor((pixelDepthVS - minDepthVS) * depthRangeRecip))));

    atomicOr(tileDepthMask, 1 << depthMaskCellIndex);

    barrier();

    AABB tileAABB = createTileAABB(minDepthZ, maxDepthZ, texSize);
    //Frustum tileFrustum = createTileFrustum(minDepthZ, maxDepthZ, texSize);
    barrier();

    cullPointLights(tileAABB, minDepthVS, depthRangeRecip);
    //cullPointLights(tileFrustum, minDepthVS, depthRangeRecip);
    cullSpotLights(tileAABB);
    cullLightProbes(tileAABB, minDepthVS, depthRangeRecip);

    barrier();

#ifdef DEBUG
    if (gl_LocalInvocationIndex == 0)
    {
        imageStore(lightCountImage, ivec2(gl_WorkGroupID.xy), vec4(float(pointLightCount), float(spotLightCount), float(lightProbesCount), 0));
        //imageStore(lightCountImage, ivec2(gl_WorkGroupID.xy), uvec4(3, 3, 3, 0));
    }
#endif
}

Frustum createTileFrustum(float minDepthZ, float maxDepthZ, vec2 texSize)
{
    //calculate tile corners (2d points in pixels)
    uint minX = MAX_WORK_GROUP_SIZE * gl_WorkGroupID.x;
    uint minY = MAX_WORK_GROUP_SIZE * gl_WorkGroupID.y;
    uint maxX = MAX_WORK_GROUP_SIZE * (gl_WorkGroupID.x + 1);
    uint maxY = MAX_WORK_GROUP_SIZE * (gl_WorkGroupID.y + 1);

    //Convert tile corners to NDC and then to view space
    vec3 tileCorners[4];
    const float infZProjectionAdjustment = 0.00001f; //protection when projection far plane z = 0 is inf
    const float depth = max(minDepthZ, infZProjectionAdjustment);
    tileCorners[0] = fromNdcToViewSpace(vec4(pixToNDC(minX, texSize.x), pixToNDC(minY, texSize.y), depth, 1.0f));
    tileCorners[1] = fromNdcToViewSpace(vec4(pixToNDC(maxX, texSize.x), pixToNDC(minY, texSize.y), depth, 1.0f));
    tileCorners[2] = fromNdcToViewSpace(vec4(pixToNDC(maxX, texSize.x), pixToNDC(maxY, texSize.y), depth, 1.0f));
    tileCorners[3] = fromNdcToViewSpace(vec4(pixToNDC(minX, texSize.x), pixToNDC(maxY, texSize.y), depth, 1.0f));

    //create the frustum planes by using the product between these points
    vec4 planes[6];
    for(int i = 0; i < 4; ++i)
    {
        planes[i].xyz = createPlane(tileCorners[i],tileCorners[(i+1) & 3]);
    }

    //create far plane
    planes[4] = createPlanePerpendicularToView(tileCorners[0] - tileCorners[1], 
        tileCorners[2] - tileCorners[1], tileCorners[1]);

    //create nearPlane
    vec3 nearTileCorners[3];
    nearTileCorners[0] = fromNdcToViewSpace(vec4(pixToNDC(minX, texSize.x), pixToNDC(minY, texSize.y), maxDepthZ, 1.0f));
    nearTileCorners[1] = fromNdcToViewSpace(vec4(pixToNDC(maxX, texSize.x), pixToNDC(minY, texSize.y), maxDepthZ, 1.0f));
    nearTileCorners[2] = fromNdcToViewSpace(vec4(pixToNDC(maxX, texSize.x), pixToNDC(maxY, texSize.y), maxDepthZ, 1.0f));

    planes[5] = createPlanePerpendicularToView(nearTileCorners[2] - nearTileCorners[1], 
        nearTileCorners[0] - nearTileCorners[1], nearTileCorners[1]);

    return Frustum(planes);
}

AABB createTileAABB(float minDepthZ, float maxDepthZ, vec2 texSize)
{
    //calculate tile corners (2d points in pixels)
    uint minX = MAX_WORK_GROUP_SIZE * gl_WorkGroupID.x;
    uint minY = MAX_WORK_GROUP_SIZE * gl_WorkGroupID.y;
    uint maxX = MAX_WORK_GROUP_SIZE * (gl_WorkGroupID.x + 1);
    uint maxY = MAX_WORK_GROUP_SIZE * (gl_WorkGroupID.y + 1);

    //Convert bottom tile corners to NDC and then to view space
    const float depth = max(minDepthZ, 0.00001f); //protection when projection far plane z = 0 is inf
    vec3 tileFarBottomLeftCorner = fromNdcToViewSpace(vec4(pixToNDC(minX, texSize.x), pixToNDC(minY, texSize.y), depth, 1.0f));
    vec3 tileFarUpperRightCorner = fromNdcToViewSpace(vec4(pixToNDC(maxX, texSize.x), pixToNDC(maxY, texSize.y), depth, 1.0f));

    vec3 tileNearBottomLeftCorner = fromNdcToViewSpace(vec4(pixToNDC(minX, texSize.x), pixToNDC(minY, texSize.y), maxDepthZ, 1.0f));
    vec3 tileNearUpperRightCorner = fromNdcToViewSpace(vec4(pixToNDC(maxX, texSize.x), pixToNDC(maxY, texSize.y), maxDepthZ, 1.0f));

    //AABB's min-max points
    float minXViewSpace = min(tileFarBottomLeftCorner.x, tileNearBottomLeftCorner.x);
    float minYViewSpace = min(tileFarBottomLeftCorner.y, tileNearBottomLeftCorner.y);
    vec3 minPoint = vec3(minXViewSpace, minYViewSpace, tileFarBottomLeftCorner.z);

    float maxXViewSpace = max(tileFarUpperRightCorner.x, tileNearUpperRightCorner.x);
    float maxYViewSpace = max(tileFarUpperRightCorner.y, tileNearUpperRightCorner.y);
    vec3 maxPoint = vec3(maxXViewSpace, maxYViewSpace, tileNearUpperRightCorner.z); 

    vec3 aabbHalfSize = abs(maxPoint - minPoint) * 0.5;
    vec3 aabbCenter = minPoint + aabbHalfSize;

    return AABB(aabbCenter, aabbHalfSize);
}

bool testSphereVsAABB(vec3 sphereCenter, float sphereRadius, vec3 AABBCenter, vec3 AABBHalfSize)
{
    vec3 delta = vec3(0);
    delta.x = max(0, abs(AABBCenter.x - sphereCenter.x) - AABBHalfSize.x);
    delta.y = max(0, abs(AABBCenter.y - sphereCenter.y) - AABBHalfSize.y);
    delta.z = max(0, abs(AABBCenter.z - sphereCenter.z) - AABBHalfSize.z);
    float distSq = dot(delta, delta);
    return distSq <= (sphereRadius * sphereRadius);
}

vec3 fromNdcToViewSpace(vec4 ndcPoint)
{
    vec4 viewSpacePosition = camera.invertedProjection * ndcPoint;
    return viewSpacePosition.xyz / viewSpacePosition.w;
}

float pixToNDC(uint point, float dimSize)
{
    return (float(point) / dimSize) * 2.0f - 1.0f; 
}

vec3 createPlane(vec3 p1, vec3 p2)
{
    //coefficients for plane equation are:
    //A = normalize(cross(p1, p2)).x;
    //B = normalize(cross(p1, p2)).y;
    //C = normalize(cross(p1, p2)).z;
    //D = 0 -> because plane goes through the point (0,0,0)
return normalize(cross(p1, p2));
}

vec4 createPlanePerpendicularToView(vec3 dir1, vec3 dir2, vec3 pointOnPlane)
{
    vec4 planeCoeffs = {0,0,0,0};
    planeCoeffs.xyz = normalize(cross(dir1, dir2));
    planeCoeffs.w = -dot(planeCoeffs.xyz, pointOnPlane);
    return planeCoeffs;
}

float getDistanceFromPlane(vec3 position, vec3 plane)
{
    return dot(position, plane);
}

float getDistanceFromPerpendicularPlane(vec3 position, vec4 perpendicularPlane)
{
    return dot(position, perpendicularPlane.xyz) + perpendicularPlane.w;
}

void cullPointLights(Frustum tileFrustum, float minDepthVS, float depthRangeRecip)
{
    //Now check the lights against the frustum and append them to the list

    for (uint i = gl_LocalInvocationIndex; i < pointLights.length(); i += THREADS_PER_TILE)
    {
        uint il = i;
        PointLight p = pointLights[il];

        vec3 lightViewPosition = (camera.view * vec4(p.positionAndRadius.xyz, 1.0f)).xyz;
        float r = p.positionAndRadius.w;

        float zMin = lightViewPosition.z - r;
        float zMax = lightViewPosition.z + r;

        uint lightMaskCellIndexStart = uint(max(0, min(32, floor(zMin - minDepthVS) * depthRangeRecip)));
        uint lightMaskCellIndexEnd = uint(max(0, min(32, floor(zMax - minDepthVS) * depthRangeRecip)));

        uint lightMask = 0xffffffffu;
        lightMask >>= 31 - (lightMaskCellIndexEnd - lightMaskCellIndexStart);
        lightMask <<= lightMaskCellIndexStart;

        bool intersect2_5D = bool(tileDepthMask & lightMask);

        if (!intersect2_5D)
            continue;

        if( 
            ( getDistanceFromPlane(lightViewPosition, tileFrustum.planes[0].xyz) < r ) &&
            ( getDistanceFromPlane(lightViewPosition, tileFrustum.planes[1].xyz) < r ) &&
            ( getDistanceFromPlane(lightViewPosition, tileFrustum.planes[2].xyz) < r ) &&
            ( getDistanceFromPlane(lightViewPosition, tileFrustum.planes[3].xyz) < r)  &&
            ( getDistanceFromPerpendicularPlane(lightViewPosition, tileFrustum.planes[4]) < r) &&
            ( getDistanceFromPerpendicularPlane(lightViewPosition, tileFrustum.planes[5]) < r)
            )
        {
            uint id = atomicAdd(pointLightCount, 1);
            pointLightIndices[lightBeginIndex + id] = il;
        }
    }

    pointLightIndices[numberOfLightsIndex] = pointLightCount;
}

void cullPointLights(AABB tileAABB, float minDepthVS, float depthRangeRecip)
{
    //Now check the lights against the frustum and append them to the list

    for (uint i = gl_LocalInvocationIndex; i < pointLights.length(); i += THREADS_PER_TILE)
    {
        uint il = i;
        PointLight p = pointLights[il];

        vec3 lightViewPosition = (camera.view * vec4(p.positionAndRadius.xyz, 1.0f)).xyz;
        float r = p.positionAndRadius.w;

        float zMin = lightViewPosition.z - r;
        float zMax = lightViewPosition.z + r;

        uint lightMaskCellIndexStart = uint(max(0, min(32, floor(zMin - minDepthVS) * depthRangeRecip)));
        uint lightMaskCellIndexEnd = uint(max(0, min(32, floor(zMax - minDepthVS) * depthRangeRecip)));

        uint lightMask = 0xffffffffu;
        lightMask >>= 31 - (lightMaskCellIndexEnd - lightMaskCellIndexStart);
        lightMask <<= lightMaskCellIndexStart;

        bool intersect2_5D = bool(tileDepthMask & lightMask);

        if (!intersect2_5D)
            continue;

        if( testSphereVsAABB(lightViewPosition, r, tileAABB.aabbCenter, tileAABB.aabbHalfSize) )
        {
            uint id = atomicAdd(pointLightCount, 1);
            pointLightIndices[lightBeginIndex + id] = il;
        }
    }

    pointLightIndices[numberOfLightsIndex] = pointLightCount;
}

void cullSpotLights(AABB tileAABB)
{
    //Now check the lights against the frustum and append them to the list

    for (uint i = gl_LocalInvocationIndex; i < spotLights.length(); i += THREADS_PER_TILE)
    {
        uint il = i;
        SpotLight s = spotLights[il];

        vec3 lightViewPosition = (camera.view * vec4(s.boundingSphere.xyz, 1.0f)).xyz;
        float r = s.boundingSphere.w;

        if( testSphereVsAABB(lightViewPosition, r, tileAABB.aabbCenter, tileAABB.aabbHalfSize) )
        {
            uint id = atomicAdd(spotLightCount, 1);
            spotLightIndices[lightBeginIndex + id] = il;
        }
    }

    spotLightIndices[numberOfLightsIndex] = spotLightCount;
}

void cullLightProbes(AABB tileAABB, float minDepthVS, float depthRangeRecip)
{
    //Now check the lights against the frustum and append them to the list

    for (uint i = gl_LocalInvocationIndex; i < lightProbes.length(); i += THREADS_PER_TILE)
    {
        LightProbe lightProbe = lightProbes[i];

        vec3 lightViewPosition = (camera.view * vec4(lightProbe.position, 1.0f)).xyz;
        float r = lightProbe.radius;

        float zMin = lightViewPosition.z - r;
        float zMax = lightViewPosition.z + r;

        uint lightMaskCellIndexStart = uint(max(0, min(32, floor(zMin - minDepthVS) * depthRangeRecip)));
        uint lightMaskCellIndexEnd = uint(max(0, min(32, floor(zMax - minDepthVS) * depthRangeRecip)));

        uint lightMask = 0xffffffffu;
        lightMask >>= 31 - (lightMaskCellIndexEnd - lightMaskCellIndexStart);
        lightMask <<= lightMaskCellIndexStart;

        bool intersect2_5D = bool(tileDepthMask & lightMask);

        if (!intersect2_5D)
            continue;

        if( testSphereVsAABB(lightViewPosition, r, tileAABB.aabbCenter, tileAABB.aabbHalfSize) )
        {
            uint id = atomicAdd(lightProbesCount, 1);
            lightProbeIndices[lightBeginIndex + id] = i;
        }
    }

    lightProbeIndices[numberOfLightsIndex] = lightProbesCount;
}