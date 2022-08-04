#type vertex
#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 textureCoords;

layout (location = 1) out vec2 texCoords;

void main()
{
    texCoords = textureCoords;
    gl_Position = vec4(position, 1);
}

#type fragment
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

layout(location = 0) out vec4 FragColor;

layout (location = 1) in vec2 texCoords;

layout(binding = 0) uniform sampler2D depthTexture;
layout(binding = 1) uniform sampler2D diffuseTexture;
layout(binding = 2) uniform sampler2D normalTexture;
layout(binding = 3) uniform sampler2D rougnessMetalnessTexture;
layout(binding = 4) uniform samplerCube irradianceCubemap;
layout(binding = 5) uniform samplerCube prefilterCubemap;
layout(binding = 6) uniform sampler2D brdfLUT;
layout(binding = 7) uniform sampler2D ssaoTexture;
layout (binding = 8) uniform samplerCubeArray lightProbeIrradianceCubemapArray;
layout (binding = 9) uniform samplerCubeArray lightProbePrefilterCubemapArray;

layout (std140, binding = 0) uniform Camera
{
    CameraData camera;
};

layout(std430, binding = 0) buffer DirLightData
{
    DirLight dirLights[];
};

layout(std430, binding = 1) buffer PointLightData
{
    PointLight pointLights[];
};

layout(std430, binding = 2) buffer SpotLightData
{
    SpotLight spotLights[];
};

layout(std430, binding = 3) readonly buffer LightProbeData
{
    LightProbe lightProbes[];
};

float calculateAttenuation(vec3 lightPos, vec3 Pos);

vec3 directionalLightAddition(vec3 V, vec3 N, Material m);
vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m);
vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m);
vec3 ambientLightAddition(vec3 V, vec3 N, vec3 P, Material m);

vec3 worldPosFromDepth(float depth) {
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

void main()
{
    float depthValue = texture(depthTexture, texCoords).x;
    if (depthValue == 0) // 0.0f means its far plane or there is nothing in G-Buffer
    {
        discard;
    }

    const vec3 albedo = texture(diffuseTexture, texCoords).rgb;
    const vec2 normal = texture(normalTexture, texCoords).xy;
    const vec2 roughnessAndMetalness = texture(rougnessMetalnessTexture, texCoords).rg;
    const float ssao = texture(ssaoTexture, texCoords).x;

    const vec3 decoded = decodeViewSpaceNormal(normal.xy);
    const vec3 worldPosNormal = (camera.invertedView * vec4(decoded, 0.0f)).xyz;

    const vec3 P = worldPosFromDepth(depthValue);

    //vec3 F0 = vec3(0.04);
    const vec3 F0 = vec3(0.16f) * pow(1.0f - roughnessAndMetalness.r, 2.0); //frostbite3 fresnel reflectance https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf page 14

    Material material = {
        albedo.xyz, //albedo in linear space
        pow(roughnessAndMetalness.r, 1.2f),
        roughnessAndMetalness.g,
        mix(F0, albedo, roughnessAndMetalness.g)
    };

    vec3 N = worldPosNormal;
    vec3 V = normalize(camera.pos.xyz - P);

    vec3 L0 = directionalLightAddition(V, N, material);
    L0 += pointLightAddition(V, N, P, material);
    L0 += spotLightAddition(V, N, P, material);
    const vec3 ambient = ambientLightAddition(V, N, P, material);

    FragColor = vec4(L0 + ambient, 1) * (1.0f - ssao);
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

    for (uint i = 0; i < pointLights.length(); ++i)
    {
        L0 += calculatePbrLighting(pointLights[i], m, V, N, Pos, NdotV);
    }
    return L0;
}

vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m)
{
    float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };
    for (uint i = 0; i < spotLights.length(); ++i)
    {
        L0 += calculatePbrLighting(spotLights[i], m, V, N, Pos, NdotV);
    }
    return L0;
}

vec3 ambientLightAddition(vec3 V, vec3 N, vec3 P, Material m)
{
    const float NdotV = max(dot(N, V), 0.0);
    vec3 ambient = vec3(0.0);
    float iblWeight = 0.0f;
    for (int i = 0; i < lightProbes.length(); ++i)
    {
        const LightProbe lightProbe = lightProbes[i];
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