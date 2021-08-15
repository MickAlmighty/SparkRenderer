#type compute
#version 450
#extension GL_ARB_bindless_texture : require

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) uniform sampler2D depthTexture; 
layout(binding = 1) uniform samplerCube irradianceMap;
layout(binding = 2) uniform samplerCube prefilterMap;
layout(binding = 3) uniform sampler2D brdfLUT;
layout(binding = 4) uniform sampler2D ssaoTexture;

layout(rgba8, binding = 0) readonly uniform image2D diffuseImage;
layout(rg16f, binding = 1) readonly uniform image2D normalImage;
layout(rg8, binding = 2) readonly uniform image2D rougnessMetalnessImage;
layout(rgba16f, binding = 3) writeonly uniform image2D lightOutput;

#define M_PI 3.14159265359 

uniform vec2 tileSize;

layout (std140) uniform Camera
{
    vec4 pos;
    mat4 view;
    mat4 projection;
    mat4 invertedView;
    mat4 invertedProjection;
    float nearZ;
    float farZ;
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

vec3 worldPosFromDepth(float depth, vec2 texCoords);
vec3 decodeViewSpaceNormal(vec2 enc);

float normalDistributionGGX(vec3 N, vec3 H, float roughness);
vec3 fresnelSchlick(vec3 V, vec3 H, vec3 F0);
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness);
float geometrySchlickGGX(float cosTheta, float k);
float geometrySmith(float NdotL, float NdotV, float roughness);
float computeSpecOcclusion(float NdotV, float AO, float roughness);
float calculateAttenuation(vec3 lightPos, vec3 Pos);
float calculateAttenuation(vec3 lightPos, vec3 pos, float maxDistance);

vec3 directionalLightAddition(vec3 V, vec3 N, Material m);
vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m, const uint clusterIndex);
vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m, const uint clusterIndex);

vec3 calculateDiffuseIBL(vec3 N, vec3 V, float NdotV, Material material, samplerCube irradianceCubemap);
vec3 calculateSpecularIBL(vec3 N, vec3 V, float NdotV, Material material);
vec4 calculateSpecularFromLightProbe(vec3 N, vec3 V, vec3 P, float NdotV, Material material, LightProbe lightProbe);
vec3 getDiffuseDominantDir(vec3 N , vec3 V , float NdotV , float roughness);
vec3 getSpecularDominantDir( vec3 N , vec3 R, float roughness );
float raySphereIntersection(vec3 rayCenter, vec3 rayDir, vec3 sphereCenter, float sphereRadius);
float computeDistanceBaseRoughness(float distInteresectionToShadedPoint, float distInteresectionToProbeCenter, float linearRoughness );

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
    const float clustersZ = 32.0f;
    const float logNearByFar = log(camera.farZ / camera.nearZ);
    const float equation3Part1 = clustersZ / logNearByFar;
    const float equation3Part2 = clustersZ * log(camera.nearZ) / logNearByFar;
    return uint(log(abs(viewSpaceDepth)) * equation3Part1 - equation3Part2);
}

uint calculateClusterIndex(vec2 screenCoords, uint clusterZ)
{
    const uint clustersX = 64;
    const uint clustersY = 64;
    uint screenSliceOffset = clustersX * clustersY * clusterZ;

    uvec2 clusterAssignmentXY = uvec2(screenCoords / tileSize);
    uint onScreenSliceIndex = clusterAssignmentXY.x * clustersY + clusterAssignmentXY.y;

    return screenSliceOffset + onScreenSliceIndex;
}

void main()
{
    const vec2 texSize = vec2(imageSize(diffuseImage));
    const ivec2 texCoords = ivec2(gl_GlobalInvocationID.xy);

    const float depthFloat = texelFetch(depthTexture, texCoords, 0).x;

    if (depthFloat == 0)
        return;

    const vec3 viewSpacePos = fromPxToViewSpace(texCoords, texSize, depthFloat);
    const uint clusterZ = getZSlice(viewSpacePos.z);
    const uint clusterIndex = calculateClusterIndex(texCoords, clusterZ);

//light calculations in world space
    const vec3 albedo = imageLoad(diffuseImage, texCoords).rgb;
    const vec2 encodedNormal = imageLoad(normalImage, texCoords).xy;
    const vec2 roughnessMetalness = imageLoad(rougnessMetalnessImage, texCoords).xy;
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
    vec3 ambient = vec3(0.0);
    float ssao = texture(ssaoTexture, vec2(texCoords) / texSize).x;
    float NdotV = max(dot(N, V), 0.0);

    float iblWeight = 0.0f;
    const uint lightProbeCount = lightIndicesBufferMetadata[clusterIndex].lightProbeCount;
    const uint globalLightProbesOffset = lightIndicesBufferMetadata[clusterIndex].lightProbeIndicesOffset;
    for(uint i = 0; i < lightProbeCount; ++i)
    {
        const uint lightProbeIndex = globalLightProbeIndices[globalLightProbesOffset + i];
        LightProbe lightProbe = lightProbes[lightProbeIndex];
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

    vec4 color = vec4(min(L0 + ambient, vec3(65000)), 1) * (1.0f - ssao);

    //barrier();
    //imageStore(lightOutput, texCoords, vec4(lightProbeIndices[numberOfLightsIndex]));
    imageStore(lightOutput, texCoords, color);
}

vec3 worldPosFromDepth(float depth, vec2 texCoords) {
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

vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m, const uint clusterIndex)
{
    float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };
    const uint pointLightCount = lightIndicesBufferMetadata[clusterIndex].pointLightCount;
    const uint globalPointLightsOffset = lightIndicesBufferMetadata[clusterIndex].pointLightIndicesOffset;
    for (int index = 0; index < pointLightCount; ++index)
    {
        const uint pointLightIndex = globalPointLightIndices[globalPointLightsOffset + index];
        PointLight p = pointLights[pointLightIndex];

        vec3 lightPos = p.positionAndRadius.xyz;
        float lightRadius = p.positionAndRadius.w;
        vec3 L = normalize(lightPos - Pos);
        vec3 H = normalize(V + L);

        float NdotL = max(dot(N, L), 0.0f);

        vec3 F = fresnelSchlick(V, H, m.F0);
        float D = normalDistributionGGX(N, H, m.roughness);
        float G = geometrySmith(NdotV, NdotL, m.roughness);
        
        vec3 radiance = p.color * calculateAttenuation(lightPos, Pos, lightRadius);

        vec3 kD = mix(vec3(1.0) - F, vec3(0.0), m.metalness);
        vec3 diffuseColor = kD * m.albedo / M_PI;

        vec3 specularColor = (F * D * G) / max(4 * NdotV * NdotL, 0.00001);

        L0 += (diffuseColor + specularColor) * radiance * NdotL;
    }

    return L0;
}

vec3 spotLightAddition(vec3 V, vec3 N, vec3 Pos, Material m, const uint clusterIndex)
{
    float NdotV = max(dot(N, V), 0.0f);

    vec3 L0 = { 0, 0, 0 };
    const uint spotLightCount = lightIndicesBufferMetadata[clusterIndex].spotLightCount;
    const uint globalSpotLightsOffset = lightIndicesBufferMetadata[clusterIndex].spotLightIndicesOffset;
    for (int index = 0; index < spotLightCount; ++index)
    {
        const uint spotLightIndex = globalSpotLightIndices[globalSpotLightsOffset + index];
        SpotLight s = spotLights[spotLightIndex];
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