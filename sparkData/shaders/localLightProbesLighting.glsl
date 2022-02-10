#type vertex
#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 textureCoords;

layout (location = 0) out vec2 texCoords;

void main()
{
    texCoords = textureCoords;
    gl_Position = vec4(position, 1);
}

#type fragment
#version 450
layout(location = 0) out vec4 FragColor;

layout (location = 0) in vec2 texCoords;
#define M_PI 3.14159265359 

layout(binding = 0) uniform sampler2D depthTexture;
layout(binding = 1) uniform sampler2D diffuseTexture;
layout(binding = 2) uniform sampler2D normalTexture;
layout(binding = 3) uniform sampler2D rougnessMetalnessTexture;

layout (std140, binding = 0) uniform Camera
{
    vec4 pos;
    mat4 view;
    mat4 projection;
    mat4 invertedView;
    mat4 invertedProjection;
} camera;

struct DirLight {
    vec3 direction;
    float nothing;
    vec3 color;
    float nothing2;
};

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

layout(std430, binding = 1) buffer DirLightData
{
    DirLight dirLights[];
};

layout(std430, binding = 2) buffer PointLightData
{
    PointLight pointLights[];
};

layout(std430, binding = 3) buffer SpotLightData
{
    SpotLight spotLights[];
};

float normalDistributionGGX(vec3 N, vec3 H, float roughness);
vec3 fresnelSchlick(vec3 V, vec3 H, vec3 F0);
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness);
float geometrySchlickGGX(float cosTheta, float k);
float geometrySmith(float NdotL, float NdotV, float roughness);
float computeSpecOcclusion(float NdotV, float AO, float roughness);
float calculateAttenuation(vec3 lightPos, vec3 Pos);
float calculateAttenuation(vec3 lightPos, vec3 pos, float maxDistance);

struct Material
{
    vec3 albedo;
    float roughness;
    float metalness;
    vec3 F0;
};

vec3 directionalLightAddition(vec3 V, vec3 N, Material m);
vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m);
vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m);

vec3 worldPosFromDepth(float depth) {
    vec4 clipSpacePosition = vec4(texCoords * 2.0 - 1.0, depth, 1.0);
    vec4 viewSpacePosition = camera.invertedProjection * clipSpacePosition;
    vec4 worldSpacePosition = camera.invertedView * viewSpacePosition;
    return worldSpacePosition.xyz /= worldSpacePosition.w; //perspective division
}

vec3 decodeViewSpaceNormal(vec2 enc)
{
    //Lambert Azimuthal Equal-Area projection
    //http://aras-p.info/texts/CompactNormalStorage.html
    vec2 fenc = enc*4-2;
    float f = dot(fenc,fenc);
    float g = sqrt(1-f/4);
    vec3 n;
    n.xy = fenc*g;
    n.z = 1-f/2;
    return n;
}

vec3 attenuateHighFrequencies(vec3 color)
{
    const float luma = dot(color, vec3(0.299, 0.587, 0.114));
    float weight = 1 / (1 + luma * 0.1f);
    return color * weight;
}

void main()
{
    float depthValue = texture(depthTexture, texCoords).x;
    if (depthValue == 0.0f) // 0.0f means its far plane or there is nothing in G-Buffer
    {
        discard;
    }
    vec3 pos = worldPosFromDepth(depthValue);

    vec3 albedo = texture(diffuseTexture, texCoords).rgb;
    vec2 normal = texture(normalTexture, texCoords).xy;
    vec2 roughnessAndMetalness = texture(rougnessMetalnessTexture, texCoords).rg;

    vec3 decoded = decodeViewSpaceNormal(normal.xy);

    vec3 worldPosNormal = (camera.invertedView * vec4(decoded, 0.0f)).xyz;

    Material material = {
        albedo.xyz, //albedo in linear space
        pow(roughnessAndMetalness.r, 1.2f), //roughness
        0.0f,   //metalness
        vec3(0.0f) // F0 reflectance
    };

    material.F0 = vec3(0.16f) * (1 - material.roughness); //frostbite3 fresnel reflectance https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf page 14
    //material.F0 = mix(F0, material.albedo, material.metalness); //metalness is equal 0

    vec3 N = worldPosNormal;
    vec3 V = normalize(camera.pos.xyz - pos);

    vec3 L0 = directionalLightAddition(V, N, material);
    L0 += pointLightAddition(V, N, pos, material);
    L0 += spotLightAddition(V, N, pos, material);

    vec4 color = vec4(L0, 1);
    FragColor = vec4(attenuateHighFrequencies(color.xyz), 1.0f);
}

vec3 directionalLightAddition(vec3 V, vec3 N, Material m)
{
    float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };
    for (uint i = 0; i < dirLights.length(); ++i)
    {
        vec3 L = normalize(-dirLights[i].direction);
        vec3 H = normalize(V + L);

        float NdotL = max(dot(N, L), 0.0f);

        vec3 F = fresnelSchlick(V, H, m.F0);
        float D = normalDistributionGGX(N, H, m.roughness);
        float G = geometrySmith(NdotL, NdotV, m.roughness);

        vec3 kD = mix(vec3(1.0) - F, vec3(0.0), m.metalness);
        vec3 diffuseColor = kD * m.albedo / M_PI;

        vec3 specularColor = (F * D * G) / max(4 * NdotV * NdotL, 0.00001);
        
        L0 += (diffuseColor + specularColor) * dirLights[i].color * NdotL;
    }
    return L0;
}

vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m)
{
    float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };

    for (uint i = 0; i < pointLights.length(); ++i)
    {
        vec3 lightPos = pointLights[i].positionAndRadius.xyz;
        float lightRadius = pointLights[i].positionAndRadius.w;
        vec3 L = normalize(lightPos - Pos);
        vec3 H = normalize(V + L);

        float NdotL = max(dot(N, L), 0.0f);

        vec3 F = fresnelSchlick(V, H, m.F0);
        float D = normalDistributionGGX(N, H, m.roughness);
        float G = geometrySmith(NdotV, NdotL, m.roughness);
        
        vec3 radiance = pointLights[i].color * calculateAttenuation(lightPos, Pos, lightRadius);

        vec3 kD = mix(vec3(1.0) - F, vec3(0.0), m.metalness);
        vec3 diffuseColor = kD * m.albedo / M_PI;

        vec3 specularColor = (F * D * G) / max(4 * NdotV * NdotL, 0.00001);

        L0 += (diffuseColor + specularColor) * radiance * NdotL;
    }
    return L0;
}

vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m)
{
    float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };
    for (uint i = 0; i < spotLights.length(); ++i)
    {
        vec3 directionToLight = normalize(-spotLights[i].direction);
        vec3 L = normalize(spotLights[i].position - Pos);

        float theta = dot(directionToLight, L);
        float epsilon = max(spotLights[i].cutOff - spotLights[i].outerCutOff, 0.0);
        float intensity = clamp((theta - spotLights[i].outerCutOff) / epsilon, 0.0, 1.0);  

        vec3 H = normalize(V + L);

        float NdotL = max(dot(N, L), 0.0f);

        vec3 F = fresnelSchlick(V, H, m.F0);
        float D = normalDistributionGGX(N, H, m.roughness);
        float G = geometrySmith(NdotV, NdotL, m.roughness);
        
        vec3 radiance = spotLights[i].color * calculateAttenuation(spotLights[i].position, Pos, spotLights[i].maxDistance);
        radiance *= intensity;

        vec3 kD = mix(vec3(1.0) - F, vec3(0.0), m.metalness);
        vec3 diffuseColor = kD * m.albedo / M_PI;

        vec3 specularColor = (F * D * G) / max(4 * NdotV * NdotL, 0.00001);

        L0 += (diffuseColor + specularColor) * radiance * NdotL;
    }
    return L0;
}

float normalDistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);

    float nom = a2;
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    const float saveValue = 0.0001f;
    return nom / max((M_PI * denom * denom), saveValue);
}

vec3 fresnelSchlick(vec3 V, vec3 H, vec3 F0)
{
    float cosTheta = max(dot(V, H), 0.0);
    return F0 + (vec3(1.0) - F0) * pow(max(1.0 - cosTheta, 0.0f), 5);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(max(1.0 - cosTheta, 0.0f), 5.0);
}

float geometrySchlickGGX(float cosTheta, float k)
{
    return cosTheta / (cosTheta * (1.0 - k) + k);
}

float geometrySmith(float NdotL, float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return geometrySchlickGGX(NdotL, k) * geometrySchlickGGX(NdotV, k);
}

float computeSpecOcclusion(float NdotV, float AO, float roughness)
{
    //https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    //page 77
    return clamp(pow(NdotV + AO, exp2(-16.0f * roughness - 1.0f)) - 1.0f + AO, 0.0f, 1.0f);
}

float calculateAttenuation(vec3 lightPos, vec3 Pos)
{
    float distance    = length(lightPos - Pos);
    float attenuation = 1.0 / (distance * distance);
    return attenuation; 
}

float calculateAttenuation(vec3 lightPos, vec3 pos, float maxDistance)
{
    //https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    //page 31
    float distance    = length(lightPos - pos);
    float squaredDistance = distance * distance;

    float invSqrAttRadius = 1 / (maxDistance * maxDistance);
    float factor = squaredDistance * invSqrAttRadius;
    float smoothFactor = clamp(1.0f - factor * factor, 0.0f, 1.0f);
    float attenuation = 1.0 / max(squaredDistance, 0.01f * 0.01f) * smoothFactor * smoothFactor;
    return attenuation;
}