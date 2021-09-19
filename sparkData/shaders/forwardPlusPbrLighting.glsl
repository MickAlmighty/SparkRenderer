#type vertex
#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texture_coords;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;

uniform mat4 model;

layout (std140) uniform Camera
{
    vec4 pos;
    mat4 view;
    mat4 projection;
    mat4 invertedView;
    mat4 invertedProjection;
} camera;

out VS_OUT {
    vec2 tex_coords;
    mat3 TBN;
    vec3 tangentFragPos;
    vec3 tangentCamPos;
    vec3 worldPos;
    vec3 worldNormal;
} vs_out;

void main()
{
    vec4 worldPosition = model * vec4(position, 1);

    mat3 normalMatrix = mat3(transpose(inverse(model)));
    vec3 T = normalize(normalMatrix * tangent);
    vec3 N = normalize(normalMatrix * normal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    mat3 inverseTBN = transpose(TBN);

    // vec3 T1 = normalize(vec3(model * vec4(tangent, 0.0)));
    // vec3 N1 = normalize(vec3(model * vec4(normal, 0.0)));
    // T1 = normalize(T1 - dot(T1, N1) * N1);
    // vec3 B1 = cross(T1, N1);
    // //vec3 B1 = normalize(vec3(model * vec4(bitangent, 0.0)));
    // mat3 TBN1 = mat3(T1, B1, N1);

    vs_out.worldPos = worldPosition.xyz;
    vs_out.tangentFragPos = inverseTBN * worldPosition.xyz;
    vs_out.tangentCamPos  = inverseTBN * camera.pos.xyz;
    vs_out.tex_coords = texture_coords;
    vs_out.TBN = TBN;
    vs_out.worldNormal = vec3(model * vec4(normal, 0));

    gl_Position = camera.projection * camera.view * worldPosition;
}

#type fragment
#version 450
#extension GL_ARB_bindless_texture : require
layout (location = 0) out vec4 FragColor;

layout (binding = 1) uniform sampler2D diffuseTexture;
layout (binding = 2) uniform sampler2D normalTexture;
layout (binding = 3) uniform sampler2D roughnessTexture;
layout (binding = 4) uniform sampler2D metalnessTexture;
layout (binding = 5) uniform sampler2D heightTexture;

layout(binding = 7) uniform samplerCube irradianceMap;
layout(binding = 8) uniform samplerCube prefilterMap;
layout(binding = 9) uniform sampler2D brdfLUT;
layout(binding = 10) uniform sampler2D ssaoTexture;

in VS_OUT {
    vec2 tex_coords;
    mat3 TBN;
    vec3 tangentFragPos;
    vec3 tangentCamPos;
    vec3 worldPos;
    vec3 worldNormal;
} vs_out;

layout (std140) uniform Camera
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

#define M_PI 3.14159265359

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
vec3 getSpecularDominantDir( vec3 N , vec3 R, float roughness);
float raySphereIntersection(vec3 rayCenter, vec3 rayDir, vec3 sphereCenter, float sphereRadius);
float computeDistanceBaseRoughness(float distInteresectionToShadedPoint, float distInteresectionToProbeCenter, float linearRoughness);

const float heightScale = 0.05f;

vec2 parallaxMapping(vec2 texCoords, vec3 viewDir)
{
    // number of depth layers
    const float minLayers = 8.0;
    const float maxLayers = 32.0;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), viewDir)));
    // calculate the size of each layer
    float layerDepth = 1.0 / numLayers;
    // depth of current layer
    float currentLayerDepth = 0.0;
    // the amount to shift the texture coordinates per layer (from vector P)
    vec2 P = viewDir.xy * heightScale;
    vec2 deltaTexCoords = P / numLayers;

    vec2  currentTexCoords = texCoords;
    float currentDepthMapValue = 1.0f - texture(heightTexture, currentTexCoords).r;

    while (currentLayerDepth < currentDepthMapValue)
    {
        // shift texture coordinates along direction of P
        currentTexCoords -= deltaTexCoords;
        // get depthmap value at current texture coordinates
        currentDepthMapValue = 1.0f - texture(heightTexture, currentTexCoords).r;
        // get depth of next layer
        currentLayerDepth += layerDepth;
    }

    vec2 prevTexCoords = currentTexCoords + deltaTexCoords;

    // get depth after and before collision for linear interpolation
    float afterDepth = currentDepthMapValue - currentLayerDepth;
    float beforeDepth = (1.0f - texture(heightTexture, prevTexCoords).r) - currentLayerDepth + layerDepth;

    // interpolation of texture coordinates
    float weight = afterDepth / (afterDepth - beforeDepth);
    vec2 finalTexCoords = prevTexCoords * weight + currentTexCoords * (1.0 - weight);

    return finalTexCoords;
}

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
        tex_coords = parallaxMapping(vs_out.tex_coords, tangentViewDir);
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
    const float saveValue = 0.00000000001f;
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