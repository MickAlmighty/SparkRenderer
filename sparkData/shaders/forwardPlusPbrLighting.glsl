#type vertex
#version 450
#include "Camera.hglsl"
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texture_coords;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;

layout (push_constant) uniform Model
{
    mat4 model;
} u_Uniforms;

layout (std140, binding = 0) uniform Camera
{
    CameraData camera;
};

layout (location = 0) out VS_OUT {
    vec2 tex_coords;
    mat3 TBN;
    vec3 tangentFragPos;
    vec3 tangentCamPos;
    vec3 worldPos;
    vec3 worldNormal;
} vs_out;

void main()
{
    vec4 worldPosition = u_Uniforms.model * vec4(position, 1);

    mat3 normalMatrix = mat3(transpose(inverse(u_Uniforms.model)));
    vec3 T = normalize(normalMatrix * tangent);
    vec3 N = normalize(normalMatrix * normal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    mat3 inverseTBN = transpose(TBN);

    vs_out.worldPos = worldPosition.xyz;
    vs_out.tangentFragPos = inverseTBN * worldPosition.xyz;
    vs_out.tangentCamPos  = inverseTBN * camera.pos.xyz;
    vs_out.tex_coords = texture_coords;
    vs_out.TBN = TBN;
    vs_out.worldNormal = vec3(u_Uniforms.model * vec4(normal, 0));

    gl_Position = camera.viewProjection * worldPosition;
}

#type fragment
#version 450
#include "Camera.hglsl"
#include "pbrLighting.hglsl"
#include "DirLight.hglsl"
#include "PointLight.hglsl"
#include "SpotLight.hglsl"
#include "Material.hglsl"
#include "Constants.hglsl"
#include "ParallaxMapping.hglsl"
#include "IBL.hglsl"

layout (location = 0) out vec4 FragColor;

layout (binding = 1) uniform sampler2D diffuseTexture;
layout (binding = 2) uniform sampler2D normalTexture;
layout (binding = 3) uniform sampler2D roughnessTexture;
layout (binding = 4) uniform sampler2D metalnessTexture;
layout (binding = 5) uniform sampler2D heightTexture;

layout(binding = 7) uniform samplerCube irradianceCubemap;
layout(binding = 8) uniform samplerCube prefilterCubemap;
layout(binding = 9) uniform sampler2D brdfLUT;
layout(binding = 10) uniform sampler2D ssaoTexture;

layout (location = 0) in VS_OUT {
    vec2 tex_coords;
    mat3 TBN;
    vec3 tangentFragPos;
    vec3 tangentCamPos;
    vec3 worldPos;
    vec3 worldNormal;
} vs_out;

layout (std140, binding = 0) uniform Camera
{
    CameraData camera;
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

vec3 accurateSRGBToLinear(vec3 sRGBColor)
{
    // https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    // page 88
    vec3 linearRGBLo = sRGBColor / 12.92f;
    vec3 linearRGBHi = pow((sRGBColor + 0.055f) / 1.055f, vec3(2.4f));
    vec3 linearRGB;
    linearRGB.x = (sRGBColor.x <= 0.04045f) ? linearRGBLo.x : linearRGBHi.x;
    linearRGB.y = (sRGBColor.y <= 0.04045f) ? linearRGBLo.y : linearRGBHi.y;
    linearRGB.z = (sRGBColor.z <= 0.04045f) ? linearRGBLo.z : linearRGBHi.z;
    return linearRGB;
}

vec3 getWorldNormal(vec2 tc)
{
    vec3 normalFromTexture = texture(normalTexture, tc).xyz;
    if (normalFromTexture.xy != vec2(0))
    {
        if (normalFromTexture.z == 0)
        {
            normalFromTexture = normalize(normalFromTexture * 2.0 - 1.0);
            vec2 nXY = normalFromTexture.xy;
            normalFromTexture.z = sqrt(1.0f - (nXY.x * nXY.x) - (nXY.y * nXY.y));
            return normalize(vs_out.TBN * normalFromTexture);
        }
        normalFromTexture = normalize(vs_out.TBN * (normalFromTexture * 2.0 - 1.0));
        return normalFromTexture;
    }
    else
    {
        return normalize(vs_out.worldNormal);
    }
}

void main()
{
    vec2 tex_coords = vs_out.tex_coords;
    if (texture(heightTexture, vs_out.tex_coords).r != 0.0)
    {
        vec3 tangentViewDir = normalize(vs_out.tangentCamPos - vs_out.tangentFragPos);
        tex_coords = parallaxMapping(vs_out.tex_coords, tangentViewDir, heightTexture);
    }

    vec3 N = getWorldNormal(tex_coords);
    vec3 P = vs_out.worldPos;

    vec3 albedo = accurateSRGBToLinear(texture(diffuseTexture, tex_coords).rgb);
    vec2 roughnessAndMetalness = vec2(texture(roughnessTexture, tex_coords).x, texture(metalnessTexture, tex_coords).x);
    float ssao = texture(ssaoTexture, gl_FragCoord.xy / vec2(textureSize(ssaoTexture, 0))).x;

    Material material = {
        albedo.xyz, //albedo in linear space
        pow(roughnessAndMetalness.r, 1.2f),
        roughnessAndMetalness.g,
        vec3(0.0f)
    };

    //vec3 F0 = vec3(0.04);
    vec3 F0 = vec3(0.16f) * pow(1.0f - material.roughness, 2.0); //frostbite3 fresnel reflectance https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf page 14
    material.F0 = mix(F0, material.albedo, material.metalness);

    vec3 V = normalize(camera.pos.xyz - P);

    vec3 L0 = directionalLightAddition(V, N, material);
    L0 += pointLightAddition(V, N, P, material);
    L0 += spotLightAddition(V, N, P, material);

    //IBL here
    vec3 ambient = vec3(0.0);
    float NdotV = max(dot(N, V), 0.0);

    vec3 diffuseIBL = calculateDiffuseIBL(N, V, NdotV, material, irradianceCubemap);
    vec3 specularIBL = calculateSpecularIBL(N, V, NdotV, material, prefilterCubemap, brdfLUT);
    ambient += (diffuseIBL + specularIBL);

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

float calculateAttenuation(vec3 lightPos, vec3 Pos)
{
    float distance    = length(lightPos - Pos);
    float attenuation = 1.0 / (distance * distance);
    return attenuation; 
}