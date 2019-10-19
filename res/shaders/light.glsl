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

layout(binding = 0) uniform sampler2D positionTexture;
layout(binding = 1) uniform sampler2D diffuseTexture;
layout(binding = 2) uniform sampler2D modelNormalTexture;
layout(binding = 3) uniform sampler2D normalTexture;
layout(binding = 4) uniform sampler2D roughnessTexture;
layout(binding = 5) uniform sampler2D metalnessTexture;

struct DirLight {
	vec3 color;
	vec3 position;
};


uniform DirLight dirLight = DirLight(vec3(1, 1, 1), vec3(0, 10, 0));
uniform vec3 camPos;

float normalDistribution(vec3 N, vec3 H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;
	float nom = a2;
	float denom = M_PI * pow(NdotH2 * (a2 - 1.0) + 1.0, 2);
	return nom / max(denom, 0.001);
}

vec3 fresnelSchlick(vec3 V, vec3 H, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - clamp(dot(H, V), 0.0, 1.0), 5);
}

float geometrySchlickGGX(vec3 V, vec3 N, float roughness)
{
	float NdotV = max(dot(N, V), 0.0);
	float r = roughness;
	float k = (r*r) / 8.0;
	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;
	return nom / denom;
}

float geometrySmith(vec3 L, vec3 V, vec3 N, float roughness)
{
	return geometrySchlickGGX(V, N, roughness) * geometrySchlickGGX(L, N, roughness);
}

void main()
{
	vec3 pos = texture(positionTexture, texCoords).xyz;
	vec3 albedo = pow(texture(diffuseTexture, texCoords).xyz, vec3(2.2));
	//vec3 albedo = texture(diffuseTexture, texCoords).xyz;
	float roughness = texture(roughnessTexture, texCoords).x;
	float metalness = texture(metalnessTexture, texCoords).x;

	vec3 V = normalize(camPos - pos);
	vec3 L = normalize(dirLight.position - pos);
	//vec3 N = normalize(texture(modelNormalTexture, texCoords).xyz);
	vec3 N = texture(normalTexture, texCoords).xyz;
	vec3 H = normalize(V + L);

	vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, metalness);

	vec3 L0 = { 0, 0, 0 };
	{
		float D = normalDistribution(N, H, roughness);
		vec3 F = fresnelSchlick(V, H, F0);
		float G = geometrySmith(L, V, N, roughness);

		float VdotN = max(dot(V, N), 0.0f);
		float LdotN = max(dot(L, N), 0.0f);

		vec3 kDiffuse = vec3(1.0);// -F;
		//kDiffuse *= 1.0 - metalness;
		vec3 diffuseColor = kDiffuse * albedo / M_PI;
		vec3 specularColor = (D * F * G) / max((4 * VdotN * LdotN), 0.001);
		L0 += (diffuseColor + specularColor) * dirLight.color * LdotN;
	}
	vec3 ambient = vec3(0.001) * albedo;
	FragColor = vec4(L0 + ambient, 1);
}