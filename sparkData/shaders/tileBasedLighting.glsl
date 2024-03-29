#type compute
#version 450
#include "pbrLighting.hglsl"
#include "DirLight.hglsl"
#include "PointLight.hglsl"
#include "SpotLight.hglsl"
#include "LightProbe.hglsl"
#include "Material.hglsl"
#include "Constants.hglsl"
#include "IBL.hglsl"
#include "Camera.hglsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 1) uniform samplerCube irradianceCubemap;
layout(binding = 2) uniform samplerCube prefilterCubemap;
layout(binding = 3) uniform sampler2D brdfLUT;
layout(binding = 4) uniform sampler2D ssaoTexture;

//G-Buffer
layout(binding = 0) uniform sampler2D depthTexture; 
layout(binding = 5) uniform sampler2D diffuseImage;
layout(binding = 6) uniform sampler2D normalImage;
layout(binding = 7) uniform sampler2D rougnessMetalnessImage;
layout (binding = 11) uniform samplerCubeArray lightProbeIrradianceCubemapArray;
layout (binding = 12) uniform samplerCubeArray lightProbePrefilterCubemapArray;

layout(rgba16f, binding = 0) writeonly uniform image2D lightOutput;
#define DEBUG
#ifdef DEBUG
    layout(rgba16f, binding = 1) uniform image2D lightCountImage;
#endif

shared uint numberOfLightsIndex;
shared uint lightBeginIndex;

#define MAX_WORK_GROUP_SIZE 16
#define THREADS_PER_TILE MAX_WORK_GROUP_SIZE * MAX_WORK_GROUP_SIZE

layout (std140, binding = 0) uniform Camera
{
    CameraData camera;
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

layout(std430, binding = 4) buffer PointLightIndices
{
    uint pointLightIndices[];
};

layout(std430, binding = 5) buffer SpotLightIndices
{
    uint spotLightIndices[];
};

layout(std430, binding = 6) buffer LightProbeIndices
{
    uint lightProbeIndices[];
};

vec3 worldPosFromDepth(float depth, vec2 texCoords);
vec3 decodeViewSpaceNormal(vec2 enc);

float calculateAttenuation(vec3 lightPos, vec3 Pos);

vec3 directionalLightAddition(vec3 V, vec3 N, Material m);
vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m);
vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m);
vec3 ambientLightAddition(vec3 V, vec3 N, vec3 P, Material m);

void main()
{
    vec2 texSize = vec2(imageSize(lightOutput));
    ivec2 texCoords = ivec2(gl_GlobalInvocationID.xy);

    if (gl_LocalInvocationIndex == 0)
    {
        const uint lightIndicesPerTileLength = 4096;
        uint nrOfElementsInColumn = lightIndicesPerTileLength * gl_NumWorkGroups.y;
        uint startIndex = nrOfElementsInColumn * gl_WorkGroupID.x + lightIndicesPerTileLength * gl_WorkGroupID.y;
        numberOfLightsIndex = startIndex;
        lightBeginIndex = startIndex + 1;
    }

    barrier();

    float depthFloat = texelFetch(depthTexture, texCoords, 0).x;

    if (depthFloat == 0)
        return;

//light calculations in world space
    const vec3 albedo = texelFetch(diffuseImage, texCoords, 0).rgb;
    const vec2 encodedNormal = texelFetch(normalImage, texCoords, 0).xy;
    const vec2 roughnessMetalness = texelFetch(rougnessMetalnessImage, texCoords, 0).xy;
    float roughness = roughnessMetalness.x;
    float metalness = roughnessMetalness.y;

    Material material = {
        albedo,
        pow(roughness, 1.2),
        metalness,
        vec3(0)
    };

    vec2 screenSpaceTexCoords = vec2(texCoords) / texSize;
    vec3 P = worldPosFromDepth(depthFloat, screenSpaceTexCoords);
    vec3 N = (camera.invertedView * vec4(decodeViewSpaceNormal(encodedNormal), 0.0f)).xyz;
    vec3 V = normalize(camera.pos.xyz - P);

    //vec3 F0 = vec3(0.04);
    vec3 F0 = vec3(0.16) * pow(1 - material.roughness, 2.0); //frostbite3 fresnel reflectance https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf page 14
    material.F0 = mix(F0, material.albedo, material.metalness);

    vec3 L0 = directionalLightAddition(V, N, material);
    L0 += pointLightAddition(V, N, P, material);
    L0 += spotLightAddition(V, N, P, material);
    const vec3 ambient = ambientLightAddition(V, N, P, material);

    const float ssao = texture(ssaoTexture, vec2(texCoords) / texSize).x;
    vec4 color = vec4(min(L0 + ambient, vec3(65000)), 1) * (1.0f - ssao);

    imageStore(lightOutput, texCoords, color);

#ifdef DEBUG
    imageStore(lightCountImage, ivec2(gl_GlobalInvocationID.xy), vec4(pointLightIndices[numberOfLightsIndex], spotLightIndices[numberOfLightsIndex], lightProbeIndices[numberOfLightsIndex], 0));
#endif
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

vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m)
{
    const float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };

    for (int index = 0; index < pointLightIndices[numberOfLightsIndex]; ++index)
    {
        PointLight p = pointLights[pointLightIndices[lightBeginIndex + index]];
        L0 += calculatePbrLighting(p, m, V, N, Pos, NdotV);
    }

    return L0;
}

vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m)
{
    float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };
    for (int index = 0; index < spotLightIndices[numberOfLightsIndex]; ++index)
    {
        SpotLight s = spotLights[spotLightIndices[lightBeginIndex + index]];
        L0 += calculatePbrLighting(s, m, V, N, Pos, NdotV);
    }
    return L0;
}

vec3 ambientLightAddition(vec3 V, vec3 N, vec3 P, Material m)
{
    const float NdotV = max(dot(N, V), 0.0);
    vec3 ambient = vec3(0.0);
    float iblWeight = 0.0f;
    for (int index = 0; index < lightProbeIndices[numberOfLightsIndex]; ++index)
    {
        const LightProbe lightProbe = lightProbes[lightProbeIndices[lightBeginIndex + index]];
        const vec3 irradiance = texture(lightProbeIrradianceCubemapArray, vec4(N, lightProbe.index)).rgb;
        vec3 diffuseIBL = calculateDiffuseIBL(NdotV, m, irradiance);
        vec4 specularIBL = calculateSpecularFromLightProbe(N, V, P, NdotV, m, lightProbe, lightProbePrefilterCubemapArray, brdfLUT);

        //calculating the the smooth light fading at the border of light probe
        float localDistance = length(P - lightProbe.positionAndRadius.xyz);
        float alpha = clamp((lightProbe.positionAndRadius.w - localDistance) / 
                    max(lightProbe.positionAndRadius.w, 0.0001f), 0.0f, 1.0f);

        float alphaAttenuation = smoothstep(0.0f, 1.0f - iblWeight, alpha);
        iblWeight += alphaAttenuation;// * specularIBL.a;

        ambient += (diffuseIBL + specularIBL.rgb) * alphaAttenuation;// * specularIBL.a;

        if (iblWeight >= 1.0f)
            break;
    }

    if (iblWeight < 1.0f)
    {
        const vec3 irradiance = texture(irradianceCubemap, N).rgb;
        vec3 diffuseIBL = calculateDiffuseIBL(NdotV, m, irradiance);
        vec3 specularIBL = calculateSpecularIBL(N, V, NdotV, m, prefilterCubemap, brdfLUT);
        ambient += (diffuseIBL + specularIBL) * (1.0f - iblWeight);
    }

    return ambient;
}

float calculateAttenuation(vec3 lightPos, vec3 Pos)
{
    float distance    = length(lightPos - Pos);
    float attenuation = 1.0 / (distance * distance);
    return attenuation; 
}