#type compute
#version 450
#include "Camera.hglsl"
#include "pbrLighting.hglsl"
#include "DirLight.hglsl"
#include "PointLight.hglsl"
#include "SpotLight.hglsl"
#include "Material.hglsl"
#include "IBL.hglsl"
#include "Constants.hglsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) uniform sampler2D depthTexture; 
layout(binding = 1) uniform samplerCube irradianceCubemap;
layout(binding = 2) uniform samplerCube prefilterCubemap;
layout(binding = 3) uniform sampler2D brdfLUT;
layout(binding = 4) uniform sampler2D ssaoTexture;

layout(binding = 5) uniform sampler2D diffuseImage;
layout(binding = 6) uniform sampler2D normalImage;
layout(binding = 7) uniform sampler2D rougnessMetalnessImage;

layout(rgba16f, binding = 0) writeonly uniform image2D lightOutput;

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

struct LightProbe {
    uvec2 irradianceCubemapHandle;
    uvec2 prefilterCubemapHandle;
    vec4 positionAndRadius;
    float fadeDistance;
    float padding1;
    float padding2;
    float padding3;
};

layout(std430, binding = 0) readonly buffer DirLightData
{
    DirLight dirLights[];
};

layout(std430, binding = 1) readonly buffer PointLightData
{
    PointLight pointLights[];
};

layout(std430, binding = 2) readonly buffer SpotLightData
{
    SpotLight spotLights[];
};

layout(std430, binding = 3) readonly buffer LightProbeData
{
    LightProbe lightProbes[];
};

layout(std430, binding = 4) readonly buffer GlobalPointLightIndices
{
    uint globalPointLightIndices[];
};

layout(std430, binding = 5) readonly buffer GlobalSpotLightIndices
{
    uint globalSpotLightIndices[];
};

layout(std430, binding = 6) readonly buffer GlobalLightProbeIndices
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

layout(std430, binding = 7) readonly buffer PerClusterGlobalLightIndicesBufferMetadata
{
    LightIndicesBufferMetadata lightIndicesBufferMetadata[];
};

vec3 worldPosFromDepth(float depth, vec2 texCoords);
vec3 decodeViewSpaceNormal(vec2 enc);

float calculateAttenuation(vec3 lightPos, vec3 Pos);

vec3 directionalLightAddition(vec3 V, vec3 N, Material m);
vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m, const uint clusterIndex);
vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m, const uint clusterIndex);

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

uint calculateClusterIndex(vec2 screenCoords, uint clusterZ)
{
    const uint clustersX = algorithmData.clusterCountX;
    const uint clustersY = algorithmData.clusterCountY;
    uint screenSliceOffset = clustersX * clustersY * clusterZ;

    uvec2 clusterAssignmentXY = uvec2(screenCoords / algorithmData.pxTileSize);
    uint onScreenSliceIndex = clusterAssignmentXY.y * clustersX + clusterAssignmentXY.x;

    return screenSliceOffset + onScreenSliceIndex;
}

void main()
{
    const vec2 texSize = vec2(imageSize(lightOutput));
    const ivec2 texCoords = ivec2(gl_GlobalInvocationID.xy);

    const float depthFloat = texelFetch(depthTexture, texCoords, 0).x;

    if (depthFloat == 0)
        return;

    const vec3 viewSpacePos = fromPxToViewSpace(texCoords, texSize, depthFloat);
    const uint clusterZ = getZSlice(viewSpacePos.z);
    const uint clusterIndex = calculateClusterIndex(texCoords, clusterZ);

//light calculations in world space
    const vec3 albedo = texelFetch(diffuseImage, texCoords, 0).rgb;
    const vec2 encodedNormal = texelFetch(normalImage, texCoords, 0).xy;
    const vec2 roughnessMetalness = texelFetch(rougnessMetalnessImage, texCoords, 0).xy;

    const float roughness = roughnessMetalness.x;
    const float metalness = roughnessMetalness.y;

    Material material = {
        albedo,
        pow(roughness, 1.2),
        metalness,
        vec3(0)
    };

    vec3 P = (camera.invertedView * vec4(viewSpacePos, 1)).xyz;
    vec3 N = (camera.invertedView * vec4(decodeViewSpaceNormal(encodedNormal), 0.0f)).xyz;
    vec3 V = normalize(camera.pos.xyz - P);

    //vec3 F0 = vec3(0.04);
    vec3 F0 = vec3(0.16) * pow(1 - material.roughness, 2.0); //frostbite3 fresnel reflectance https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf page 14
    material.F0 = mix(F0, material.albedo, material.metalness);

    vec3 L0 = directionalLightAddition(V, N, material);
    L0 += pointLightAddition(V, N, P, material, clusterIndex);
    L0 += spotLightAddition(V, N, P, material, clusterIndex);

    //IBL here
    float ssao = texture(ssaoTexture, vec2(texCoords) / texSize).x;
    float NdotV = max(dot(N, V), 0.0);

    vec3 diffuseIBL = calculateDiffuseIBL(N, V, NdotV, material, irradianceCubemap);
    vec3 specularIBL = calculateSpecularIBL(N, V, NdotV, material, prefilterCubemap, brdfLUT);
    vec3 ambient = (diffuseIBL + specularIBL);

    vec4 color = vec4(min(L0 + ambient, vec3(65000)), 1) * (1.0f - ssao);

    //barrier();
    //imageStore(lightOutput, texCoords, vec4(lightIndicesBufferMetadata[clusterIndex].pointLightCount, color.yzw));
    imageStore(lightOutput, texCoords, color);
}

vec3 worldPosFromDepth(float depth, vec2 texCoords) {
    vec4 clipSpacePosition = vec4(texCoords * 2.0 - 1.0, depth, 1.0);
    vec4 worldSpacePosition = camera.invertedViewProjection * clipSpacePosition;
    return worldSpacePosition.xyz /= worldSpacePosition.w; //perspective division
}

vec3 decodeViewSpaceNormal(vec2 enc)
{
    //Lambert Azimuthal Equal-Area projection
    //http://aras-p.info/texts/CompactNormalStorage.html
    const vec2 fenc = enc*4.0f-2.0f;
    const float f = dot(fenc,fenc);
    const float g = sqrt(1.0f - (f * 0.25f));
    vec3 n;
    n.xy = fenc * g;
    n.z = 1.0f - (f * 0.5f);
    return n;
}

vec3 directionalLightAddition(vec3 V, vec3 N, Material m)
{
    const float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };
    for (uint i = 0; i < dirLights.length(); ++i)
    {
        L0 += calculatePbrLighting(dirLights[i], m, V, N, NdotV);
    }
    return L0;
}

vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m, const uint clusterIndex)
{
    const float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };
    const uint pointLightCount = lightIndicesBufferMetadata[clusterIndex].pointLightCount;
    const uint globalPointLightsOffset = lightIndicesBufferMetadata[clusterIndex].pointLightIndicesOffset;
    for (int i = 0; i < pointLightCount; ++i)
    {
        const uint index = globalPointLightIndices[globalPointLightsOffset + i];
        L0 += calculatePbrLighting(pointLights[index], m, V, N, Pos, NdotV);
    }

    return L0;
}

vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m, const uint clusterIndex)
{
    float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };
    const uint spotLightCount = lightIndicesBufferMetadata[clusterIndex].spotLightCount;
    const uint globalSpotLightsOffset = lightIndicesBufferMetadata[clusterIndex].spotLightIndicesOffset;
    for (int i = 0; i < spotLightCount; ++i)
    {
        const uint index = globalSpotLightIndices[globalSpotLightsOffset + i];
        L0 += calculatePbrLighting(spotLights[index], m, V, N, Pos, NdotV);
    }
    return L0;
}

float calculateAttenuation(vec3 lightPos, vec3 Pos)
{
    float distance    = length(lightPos - Pos);
    float attenuation = 1.0 / (distance * distance);
    return attenuation; 
}