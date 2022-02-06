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
    mat4 viewProjection;
    mat4 invertedViewProjection;
    float nearZ;
    float farZ;
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

    vs_out.worldPos = worldPosition.xyz;
    vs_out.tangentFragPos = inverseTBN * worldPosition.xyz;
    vs_out.tangentCamPos  = inverseTBN * camera.pos.xyz;
    vs_out.tex_coords = texture_coords;
    vs_out.TBN = TBN;
    vs_out.worldNormal = vec3(model * vec4(normal, 0));

    gl_Position = camera.viewProjection * worldPosition;
}

#type fragment
#version 450
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
    mat4 viewProjection;
    mat4 invertedViewProjection;
    float nearZ;
    float farZ;
} camera;

layout (std140) uniform AlgorithmData
{
    vec2 pxTileSize;
    uint clusterCountX;
    uint clusterCountY;
    uint clusterCountZ;
    float equation3Part1;
    float equation3Part2;
} algorithmData;

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

layout(std430) readonly buffer DirLightData
{
    DirLight dirLights[];
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

layout(std430) readonly buffer GlobalPointLightIndices
{
    uint globalPointLightIndices[];
};

layout(std430) readonly buffer GlobalSpotLightIndices
{
    uint globalSpotLightIndices[];
};

layout(std430) readonly buffer GlobalLightProbeIndices
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

layout(std430) readonly buffer PerClusterGlobalLightIndicesBufferMetadata
{
    LightIndicesBufferMetadata lightIndicesBufferMetadata[];
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
vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m, uint clusterIndex);
vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m, uint clusterIndex);

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

    const float viewSpaceDepth = (camera.view * vec4(vs_out.worldPos, 1.0f)).z;
    const uint clusterZ = getZSlice(viewSpaceDepth);
    uint clusterIndex = calculateClusterIndex(gl_FragCoord.xy - vec2(0.5f), clusterZ);

    vec3 N = getWorldNormal(tex_coords);
    vec3 P = vs_out.worldPos;

    vec3 albedo = accurateSRGBToLinear(texture(diffuseTexture, tex_coords).rgb);
    vec2 roughnessAndMetalness = vec2(texture(roughnessTexture, tex_coords).x, texture(metalnessTexture, tex_coords).x);
    float ssao = texture(ssaoTexture, (gl_FragCoord.xy  - vec2(0.5f)) / textureSize(ssaoTexture, 0).xy).x;

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
    L0 += pointLightAddition(V, N, P, material, clusterIndex);
    L0 += spotLightAddition(V, N, P, material, clusterIndex);

    //IBL here
    float NdotV = max(dot(N, V), 0.0);
    vec3 diffuseIBL = calculateDiffuseIBL(N, V, NdotV, material, irradianceMap);
    vec3 specularIBL = calculateSpecularIBL(N, V, NdotV, material);
    vec3 ambient = (diffuseIBL + specularIBL);

    FragColor = vec4(L0 + ambient, 1) * (1.0f - ssao);
}

vec3 directionalLightAddition(vec3 V, vec3 N, Material m)
{
    const float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };
    for (uint i = 0; i < dirLights.length(); ++i)
    {
        const vec3 L = normalize(-dirLights[i].direction);
        const vec3 H = normalize(V + L);

        const float NdotL = dot(N, L);
        if (NdotL > 0.0f)
        {
            const vec3 F = fresnelSchlick(V, H, m.F0);
            const float D = normalDistributionGGX(N, H, m.roughness);
            const float G = geometrySmith(NdotL, NdotV, m.roughness);

            const vec3 kD = mix(vec3(1.0) - F, vec3(0.0), m.metalness);
            const vec3 diffuseColor = kD * m.albedo / M_PI;

            const vec3 specularColor = (F * D * G) / max(4 * NdotV * NdotL, 0.00001);

            L0 += (diffuseColor + specularColor) * dirLights[i].color * NdotL;
        }
    }
    return L0;
}

vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m, uint clusterIndex)
{
    const float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };
    const uint pointLightCount = lightIndicesBufferMetadata[clusterIndex].pointLightCount;
    const uint globalPointLightsOffset = lightIndicesBufferMetadata[clusterIndex].pointLightIndicesOffset;
    for (int i = 0; i < pointLightCount; ++i)
    {
        const uint index = globalPointLightIndices[globalPointLightsOffset + i];
        const PointLight p = pointLights[index];

        const vec3 lightPos = p.positionAndRadius.xyz;
        const float lightRadius = p.positionAndRadius.w;
        const vec3 L = normalize(lightPos - Pos);
        const vec3 H = normalize(V + L);

        const float NdotL = dot(N, L);
        if (NdotL > 0.0f)
        {
            const vec3 F = fresnelSchlick(V, H, m.F0);
            const float D = normalDistributionGGX(N, H, m.roughness);
            const float G = geometrySmith(NdotV, NdotL, m.roughness);
            
            const vec3 radiance = p.color * calculateAttenuation(lightPos, Pos, lightRadius);

            const vec3 kD = mix(vec3(1.0) - F, vec3(0.0), m.metalness);
            const vec3 diffuseColor = kD * m.albedo / M_PI;

            const vec3 specularColor = (F * D * G) / max(4 * NdotV * NdotL, 0.00001);

            L0 += (diffuseColor + specularColor) * radiance * NdotL;
        }
    }

    return L0;
}

vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m, uint clusterIndex)
{
    float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };
    const uint spotLightCount = lightIndicesBufferMetadata[clusterIndex].spotLightCount;
    const uint globalSpotLightsOffset = lightIndicesBufferMetadata[clusterIndex].spotLightIndicesOffset;
    for (int i = 0; i < spotLightCount; ++i)
    {
        const uint index = globalSpotLightIndices[globalSpotLightsOffset + i];
        SpotLight s = spotLights[index];
        vec3 directionToLight = normalize(-s.direction);
        vec3 L = normalize(s.position - Pos);

        float theta = dot(directionToLight, L);
        float epsilon = max(s.cutOff - s.outerCutOff, 0.0);
        float intensity = clamp((theta - s.outerCutOff) / epsilon, 0.0, 1.0);  

        vec3 H = normalize(V + L);

        float NdotL = max(dot(N, L), 0.0f);

        vec3 F = fresnelSchlick(V, H, m.F0);
        float D = normalDistributionGGX(N, H, m.roughness);
        float G = geometrySmith(NdotV, NdotL, m.roughness);
        
        vec3 radiance = s.color * calculateAttenuation(s.position, Pos, s.maxDistance);
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