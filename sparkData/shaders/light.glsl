#type vertex
#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 textureCoords;

out vec2 texCoords;

void main()
{
    texCoords = textureCoords;
    gl_Position = vec4(position, 1);
}

#type fragment
#version 450
#extension GL_ARB_bindless_texture : require
layout(location = 0) out vec4 FragColor;

in vec2 texCoords;
#define M_PI 3.14159265359
#define ONE_BY_M_PI 0.31830988618

layout(binding = 0) uniform sampler2D depthTexture;
layout(binding = 1) uniform sampler2D diffuseTexture;
layout(binding = 2) uniform sampler2D normalTexture;
layout(binding = 3) uniform sampler2D rougnessMetalnessTexture;
layout(binding = 4) uniform samplerCube irradianceMap;
layout(binding = 5) uniform samplerCube prefilterMap;
layout(binding = 6) uniform sampler2D brdfLUT;
layout(binding = 7) uniform sampler2D ssaoTexture;

layout (std140) uniform Camera
{
    vec4 pos;
    mat4 view;
    mat4 projection;
    mat4 invertedView;
    mat4 invertedProjection;
    mat4 viewProjection;
    mat4 invertedViewProjection;
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

struct LightProbe {
    uvec2 irradianceCubemapHandle;
    uvec2 prefilterCubemapHandle;
    vec4 positionAndRadius;
    float fadeDistance;
    float padding1;
    float padding2;
    float padding3;
};

layout(std430) buffer DirLightData
{
    DirLight dirLights[];
};

layout(std430) buffer PointLightData
{
    PointLight pointLights[];
};

layout(std430) buffer SpotLightData
{
    SpotLight spotLights[];
};

layout(std430) readonly buffer LightProbeData
{
    LightProbe lightProbes[];
};

struct Material
{
    vec3 albedo;
    float roughness;
    float metalness;
    vec3 F0;
};

float normalDistributionGGX(vec3 N, vec3 H, float roughness);
vec3 fresnelSchlick(vec3 V, vec3 H, vec3 F0);
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness);
float geometrySchlickGGX(float cosTheta, float k);
float geometrySmith(float NdotL, float NdotV, float roughness);
float computeSpecOcclusion(float NdotV, float AO, float roughness);
float calculateAttenuation(vec3 lightPos, vec3 Pos);
float calculateAttenuation(vec3 lightPos, vec3 pos, float maxDistance);

vec3 directionalLightAddition(vec3 V, vec3 N, Material m);
vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m);
vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m);

vec3 calculateDiffuseIBL(vec3 N, vec3 V, float NdotV, Material material, samplerCube irradianceCubemap);
vec3 calculateSpecularIBL(vec3 N, vec3 V, float NdotV, Material material);
vec4 calculateSpecularFromLightProbe(vec3 N, vec3 V, vec3 P, float NdotV, Material material, LightProbe lightProbe);
vec3 getDiffuseDominantDir(vec3 N , vec3 V , float NdotV , float roughness);
vec3 getSpecularDominantDir( vec3 N , vec3 R, float roughness );
float raySphereIntersection(vec3 rayCenter, vec3 rayDir, vec3 sphereCenter, float sphereRadius);
float computeDistanceBaseRoughness(float distInteresectionToShadedPoint, float distInteresectionToProbeCenter, float linearRoughness );

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

    //IBL here
    vec3 ambient = vec3(0.0);
    float NdotV = max(dot(N, V), 0.0);

    float iblWeight = 0.0f;
    for(uint i = 0; i < lightProbes.length(); ++i)
    {
        LightProbe lightProbe = lightProbes[i];
        samplerCube irradianceSampler = samplerCube(lightProbe.irradianceCubemapHandle);
        vec3 diffuseIBL = calculateDiffuseIBL(N, V, NdotV, material, irradianceSampler);
        vec4 specularIBL = calculateSpecularFromLightProbe(N, V, P, NdotV, material, lightProbe);

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
        vec3 diffuseIBL = calculateDiffuseIBL(N, V, NdotV, material, irradianceMap);
        vec3 specularIBL = calculateSpecularIBL(N, V, NdotV, material);
        ambient += (diffuseIBL + specularIBL) * (1.0f - iblWeight);
    }

    FragColor = vec4(L0 + ambient, 1) * (1.0f - ssao);
}

vec3 directionalLightAddition(vec3 V, vec3 N, Material m)
{
    const float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };
    for (uint i = 0; i < dirLights.length(); ++i)
    {
        const vec3 L = -dirLights[i].direction;
        const vec3 H = normalize(V + L);

        const float NdotL = max(dot(N, L), 0.0f);

        if (NdotL > 0.0f)
        {
            const vec3 F = fresnelSchlick(V, H, m.F0);
            const float D = normalDistributionGGX(N, H, m.roughness);
            const float G = geometrySmith(NdotL, NdotV, m.roughness);

            const vec3 kD = mix(vec3(1.0) - F, vec3(0.0), m.metalness);
            const vec3 diffuseColor = kD * m.albedo * ONE_BY_M_PI;

            const vec3 specularColor = (F * D * G) / max(4 * NdotV * NdotL, 0.00001);

            L0 += (diffuseColor + specularColor) * dirLights[i].color * NdotL;
        }
    }
    return L0;
}

vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m)
{
    const float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };

    for (uint i = 0; i < pointLights.length(); ++i)
    {
        const vec3 lightPos = pointLights[i].positionAndRadius.xyz;
        const float lightRadius = pointLights[i].positionAndRadius.w;
        const vec3 L = normalize(lightPos - Pos);
        float NdotL = max(dot(N, L), 0.0f);
        if (NdotL > 0.0f)
        {
            const vec3 H = normalize(V + L);
            const vec3 F = fresnelSchlick(V, H, m.F0);
            const float D = normalDistributionGGX(N, H, m.roughness);
            const float G = geometrySmith(NdotV, NdotL, m.roughness);

            const vec3 radiance = pointLights[i].color * calculateAttenuation(lightPos, Pos, lightRadius);

            const vec3 kD = mix(vec3(1.0) - F, vec3(0.0), m.metalness);
            const vec3 diffuseColor = kD * m.albedo * ONE_BY_M_PI;

            const vec3 specularColor = (F * D * G) / max(4 * NdotV * NdotL, 0.00001);

            L0 += (diffuseColor + specularColor) * radiance * NdotL;
        }
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
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float NdotH = max(dot(N, H), 0.0);

    const float nom = a2;
    const float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    const float saveValue = 0.0001f;
    return nom / max((M_PI * denom * denom), saveValue);
}

vec3 fresnelSchlick(vec3 V, vec3 H, vec3 F0)
{
    const float cosTheta = max(dot(V, H), 0.0);
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
    const float r = (roughness + 1.0);
    const float k = (r * r) * 0.125; // div by 8
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
    float distance = length(lightPos - pos);
    float squaredDistance = distance * distance;

    float factor = squaredDistance / (maxDistance * maxDistance);
    float smoothFactor = clamp(1.0f - factor * factor, 0.0f, 1.0f);
    float attenuation = 1.0f / max(squaredDistance, 0.01f * 0.01f) * smoothFactor * smoothFactor;
    return attenuation; 
}

vec3 calculateDiffuseIBL(vec3 N, vec3 V, float NdotV, Material material, samplerCube irradianceCubemap)
{
    vec3 kS = fresnelSchlickRoughness(NdotV, material.F0, material.roughness);
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - material.metalness;
    vec3 irradiance = texture(irradianceCubemap, N).rgb;
    vec3 diffuse = kD * irradiance * material.albedo;
    return diffuse;
}

vec3 calculateSpecularIBL(vec3 N, vec3 V, float NdotV, Material material)
{
    vec3 R = reflect(-V, N);
    vec3 F = fresnelSchlickRoughness(NdotV, material.F0, material.roughness);
    const float MAX_REFLECTION_LOD = 4.0;

    //float mipMapLevel = material.roughness * MAX_REFLECTION_LOD; //base
    float mipMapLevel = sqrt(material.roughness * MAX_REFLECTION_LOD); //frostbite 3
    vec3 prefilteredColor = textureLod(prefilterMap, R, mipMapLevel).rgb;    
    vec2 brdf = texture(brdfLUT, vec2(NdotV, material.roughness)).rg;
        
    //float specOcclusion = computeSpecOcclusion(NdotV, ssao, material.roughness);
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y); //* specOcclusion;

    return specular;
}

vec4 calculateSpecularFromLightProbe(vec3 N, vec3 V, vec3 P, float NdotV, Material material, LightProbe lightProbe)
{
    vec4 specular = vec4(0);
    vec3 R = reflect(-V, N);
    vec3 dominantR = getSpecularDominantDir(N, R, material.roughness);

    float distToFarIntersection = raySphereIntersection(P, R, lightProbe.positionAndRadius.xyz, lightProbe.positionAndRadius.w);
    if (distToFarIntersection != 0.0f)
    {
        // Compute the actual direction to sample , only consider far intersection
        // No need to normalize for fetching cubemap
        vec3 localR = (P + distToFarIntersection * dominantR) - lightProbe.positionAndRadius.xyz;

        // We use normalized R to calc the intersection , thus intersections .y is
        // the distance between the intersection and the receiving pixel
        float distanceReceiverIntersection = distToFarIntersection;
        float distanceSphereCenterIntersection = length ( localR );

        // Compute the modified roughness based on the travelled distance
        float localRoughness = computeDistanceBaseRoughness(distanceReceiverIntersection, 
            distanceSphereCenterIntersection, material.roughness);

        const float MAX_REFLECTION_LOD = 4.0;

        //float mipMapLevel = material.roughness * MAX_REFLECTION_LOD; //base
        float mipMapLevel = sqrt(localRoughness * MAX_REFLECTION_LOD); //frostbite 3
        samplerCube prefilterSampler = samplerCube(lightProbe.prefilterCubemapHandle);
        vec4 prefilteredColor = textureLod(prefilterSampler, localR, mipMapLevel).rgba;
        vec2 brdf = texture(brdfLUT, vec2(NdotV, material.roughness)).rg;

        //float specOcclusion = computeSpecOcclusion(NdotV, ssao, material.roughness);
        vec3 F = fresnelSchlickRoughness(NdotV, material.F0, material.roughness);
        specular = vec4(prefilteredColor.rgb * (F * brdf.x + brdf.y), prefilteredColor.a);
    }

    return specular;
}

vec3 getDiffuseDominantDir(vec3 N , vec3 V , float NdotV , float roughness)
{
    float a = 1.02341f * roughness - 1.51174f;
    float b = -0.511705f * roughness + 0.755868f;
    float lerpFactor = clamp(( NdotV * a + b) * roughness, 0.0, 1.0);
    // The result is not normalized as we fetch in a cubemap
    return mix (N , V , lerpFactor);
}

// N is the normal direction
// R is the mirror vector
// This approximation works fine for G smith correlated and uncorrelated
vec3 getSpecularDominantDir( vec3 N , vec3 R, float roughness )
{
    float smoothness = clamp(0.0f, 1.0f, 1.0f - roughness);
    float lerpFactor = smoothness * ( sqrt ( smoothness ) + roughness );
    // The result is not normalized as we fetch in a cubemap
    return mix(N , R , lerpFactor);
}

float raySphereIntersection(vec3 rayCenter, vec3 rayDir, vec3 sphereCenter, float sphereRadius)
{
    vec3 rayToSphere = sphereCenter - rayCenter;
    //formula A + dot(AP,AB) / dot(AB,AB) * AB, where P is a point
    //we don't need to divide by dot(AB,AB) because rayDir is normalized then dot product is equal 1
    vec3 pointOnRayLine = rayCenter + dot(rayToSphere, rayDir) * rayDir;

    vec3 fromSphereToLine = sphereCenter - pointOnRayLine;
    float distanceSquaredToLine = dot(fromSphereToLine, fromSphereToLine);
    float radiusSquared = sphereRadius * sphereRadius;
    
    if (distanceSquaredToLine > radiusSquared)
        return 0.0f;

    if (distanceSquaredToLine == 0.0)
        return length(rayToSphere) + sphereRadius;

    float distFromProjectedPointToIntersection = sqrt(radiusSquared - distanceSquaredToLine);
    
    return length(pointOnRayLine - rayCenter) + distFromProjectedPointToIntersection;
}

float computeDistanceBaseRoughness (
    float distInteresectionToShadedPoint,
    float distInteresectionToProbeCenter,
    float linearRoughness )
{
    // To avoid artifacts we clamp to the original linearRoughness
    // which introduces an acceptable bias and allows conservation
    // of mirror reflection behavior for a smooth surface .
    float newLinearRoughness = clamp( distInteresectionToShadedPoint /
        distInteresectionToProbeCenter * linearRoughness, 0, linearRoughness);
    return mix( newLinearRoughness , linearRoughness , linearRoughness );
}