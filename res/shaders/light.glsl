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
layout(location = 0) out vec4 FragColor;

in vec2 texCoords;
#define M_PI 3.14159265359 

layout(binding = 0) uniform sampler2D depthTexture;
layout(binding = 1) uniform sampler2D diffuseTexture;
layout(binding = 2) uniform sampler2D normalTexture;
layout(binding = 3) uniform samplerCube irradianceMap;
layout(binding = 4) uniform samplerCube prefilterMap;
layout(binding = 5) uniform sampler2D brdfLUT;
layout(binding = 6) uniform sampler2D ssaoTexture;

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
	vec3 position;
	float nothing;
	vec3 color;
	float nothing2;
};

struct SpotLight {
	vec3 position;
	float cutOff;
	vec3 color;
	float outerCutOff;
	vec3 direction;
};

layout(std430) buffer DirLightData
{
	DirLight dirLights[];
} dirLightData;

layout(std430) buffer PointLightData
{
	PointLight pointLights[];
} pointLightData;

layout(std430) buffer SpotLightData
{
	SpotLight spotLights[];
} spotLightData;

float normalDistributionGGX(vec3 N, vec3 H, float roughness);
vec3 fresnelSchlick(vec3 V, vec3 H, vec3 F0);
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness);
float geometrySchlickGGX(float cosTheta, float k);
float geometrySmith(float NdotL, float NdotV, float roughness);
float calculateAttenuation(vec3 lightPos, vec3 Pos);

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

vec3 decode(vec2 enc)
{
	vec2 fenc = enc*4-2;
    float f = dot(fenc,fenc);
    float g = sqrt(1-f/4);
    vec3 n;
    n.xy = fenc*g;
    n.z = 1-f/2;
    return n;
}

void main()
{
	//vec3 pos = texture(positionTexture, texCoords).xyz;
	float depthValue = texture(depthTexture, texCoords).x;
	if (depthValue == 0.0f)
	{
		discard;
	}
	vec3 pos = worldPosFromDepth(depthValue);

	vec4 colorAndRoughness = texture(diffuseTexture, texCoords);
	vec3 normalAndMetalness = texture(normalTexture, texCoords).xyz;
	vec3 decoded = decode(normalAndMetalness.xy);
	
	vec3 worldPosNormal = (camera.invertedView * vec4(decoded, 0.0f)).xyz;
	
	Material material = {
		pow(colorAndRoughness.xyz, vec3(2.2)),
		colorAndRoughness.w,
		normalAndMetalness.z,
		vec3(0)
	};

	vec3 F0 = vec3(0.04);
	material.F0 = mix(F0, material.albedo, material.metalness);

	vec3 N = worldPosNormal;
	vec3 V = normalize(camera.pos.xyz - pos);

	vec3 L0 = directionalLightAddition(V, N, material);
	L0 += pointLightAddition(V, N, pos, material);
	L0 += spotLightAddition(V, N, pos, material);

	//IBL	
	vec3 ambient = vec3(0.0);
	{
		float NdotV = max(dot(N, V), 0.0);
		vec3 kS = fresnelSchlickRoughness(NdotV, material.F0, material.roughness);
		vec3 kD = 1.0 - kS;
		kD *= 1.0 - material.metalness;
		vec3 irradiance = texture(irradianceMap, N).rgb;
		vec3 diffuse    = irradiance * material.albedo;

		vec3 R = reflect(-V, N);
		vec3 F = fresnelSchlickRoughness(NdotV, material.F0, material.roughness);
		const float MAX_REFLECTION_LOD = 4.0;

		vec3 prefilteredColor = textureLod(prefilterMap, R, material.roughness * MAX_REFLECTION_LOD).rgb;    
		vec2 brdf = texture(brdfLUT, vec2(NdotV, material.roughness)).rg;
		vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);
		
		ambient = (kD * diffuse + specular);
	}
	float ssao = texture(ssaoTexture, texCoords).x;
	FragColor = vec4(L0 + ambient, 1) * ssao;
}

vec3 directionalLightAddition(vec3 V, vec3 N, Material m)
{
	float NdotV = max(dot(N, V), 0.0f);

	vec3 L0 = { 0, 0, 0 };
	for (uint i = 0; i < dirLightData.dirLights.length(); ++i)
	{
		vec3 L = normalize(-dirLightData.dirLights[i].direction);
		vec3 H = normalize(V + L);

		float NdotL = max(dot(N, L), 0.0f);

		vec3 F = fresnelSchlick(V, H, m.F0);
		float D = normalDistributionGGX(N, H, m.roughness);
		float G = geometrySmith(NdotL, NdotV, m.roughness);

		vec3 kD = mix(vec3(1.0) - F, vec3(0.0), m.metalness);
		vec3 diffuseColor = kD * m.albedo / M_PI;

		vec3 specularColor = (F * D * G) / max(4 * NdotV * NdotL, 0.00001);
		L0 += (diffuseColor + specularColor) * dirLightData.dirLights[i].color * NdotL;
	}
	return L0;
}

vec3 pointLightAddition(vec3 V, vec3 N, vec3 Pos, Material m)
{
	float NdotV = max(dot(N, V), 0.0f);

	vec3 L0 = { 0, 0, 0 };
	for (uint i = 0; i < pointLightData.pointLights.length(); ++i)
	{
		vec3 L = normalize(pointLightData.pointLights[i].position - Pos);
		vec3 H = normalize(V + L);

		float NdotL = max(dot(N, L), 0.0f);

		vec3 F = fresnelSchlick(V, H, m.F0);
		float D = normalDistributionGGX(N, H, m.roughness);
		float G = geometrySmith(NdotV, NdotL, m.roughness);
		
		vec3 radiance = pointLightData.pointLights[i].color * calculateAttenuation(pointLightData.pointLights[i].position, Pos);

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
	for (uint i = 0; i < spotLightData.spotLights.length(); ++i)
	{
		vec3 directionToLight = normalize(-spotLightData.spotLights[i].direction);
		vec3 L = normalize(spotLightData.spotLights[i].position - Pos);

		float theta = dot(directionToLight, L);
		float epsilon = max(spotLightData.spotLights[i].cutOff - spotLightData.spotLights[i].outerCutOff, 0.0);
		float intensity = clamp((theta - spotLightData.spotLights[i].outerCutOff) / epsilon, 0.0, 1.0);  

		vec3 H = normalize(V + L);

		float NdotL = max(dot(N, L), 0.0f);

		vec3 F = fresnelSchlick(V, H, m.F0);
		float D = normalDistributionGGX(N, H, m.roughness);
		float G = geometrySmith(NdotV, NdotL, m.roughness);
		
		vec3 radiance = spotLightData.spotLights[i].color * calculateAttenuation(spotLightData.spotLights[i].position, Pos);
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
	return nom / (M_PI * denom * denom);
}

vec3 fresnelSchlick(vec3 V, vec3 H, vec3 F0)
{
	float cosTheta = max(dot(V, H), 0.0);
	return F0 + (vec3(1.0) - F0) * pow(1.0 - cosTheta, 5);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
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

float calculateAttenuation(vec3 lightPos, vec3 Pos)
{
	float distance    = length(lightPos - Pos);
    float attenuation = 1.0 / (distance * distance);
    return attenuation; 
}