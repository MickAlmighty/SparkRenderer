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
layout (location = 0) out vec4 FragColor;

in vec2 texCoords;
#define M_PI 3.14159265359 

layout (binding = 0) uniform sampler2D positionTexture;
layout (binding = 1) uniform sampler2D diffuseTexture;
layout (binding = 2) uniform sampler2D normalTexture;
layout (binding = 3) uniform sampler2D roughnessTexture;
layout (binding = 4) uniform sampler2D metalnessTexture;

struct DirLight {
    vec3 color;
    vec3 position;
};


uniform DirLight dirLight = DirLight(vec3(1, 1, 1), vec3(0, 10, 0));
uniform vec3 camPos;

float normalDistribution(vec3 N, vec3 H, float alfa)
{
	float NdotH = max(dot(N, H), 0.0f);
    float alfaSquared = pow(alfa, 2);
    float nom = alfaSquared;
    float denom = M_PI * pow( (pow(NdotH, 2) * (alfaSquared - 1) + 1) , 2);
    return nom / max(denom, 0.001);
}

float fresnelSchlick(vec3 V, vec3 H, float metalness)
{
    float F0 = metalness;
    return F0 + (1 - F0) * pow(1 - clamp(dot(H, V), 0.0, 1.0), 5);
}

float geometrySchlickGGX(vec3 V, vec3 N, float alfa)
{
    float NdotV = max(dot(N, V), 0.0);
    float k = alfa * 0.5;
    float nom = NdotV;
    float denom = NdotV * (1 - k) + k;
    return nom / denom;
}

float geometrySmith(vec3 L, vec3 V, vec3 N, float alfa)
{
    return geometrySchlickGGX(V, N, alfa) * geometrySchlickGGX(L, N, alfa);
}

void main()
{
    vec3 pos = texture(positionTexture, texCoords).xyz;
    vec3 albedo = texture(diffuseTexture, texCoords).xyz;
    float roughness = texture(roughnessTexture, texCoords).x;
    float metalness = texture(metalnessTexture, texCoords).x;

    float A = roughness * roughness; // alfa
    vec3 V = normalize(camPos - pos);
    vec3 L = normalize( dirLight.position - pos);
    vec3 N = normalize(texture(normalTexture, texCoords).xyz);
    vec3 H = normalize(V + L);

    vec3 L0 = {0, 0, 0};
    {
        float D = normalDistribution(N, H, A);
        float F = fresnelSchlick(V, H, metalness);
        float G = geometrySmith(L, V, N, A); 

        float VdotN = max(dot(V, N), 0.0f);
        float LdotN = max(dot(L, N), 0.0f);

		float kDiffuse = 1.0;// -F;
		//kDiffuse *= 1.0 - metalness;
        vec3 diffuseColor =  kDiffuse * albedo / M_PI;
        float specularColor = (D * F * G) / max((4 * VdotN * LdotN), 0.001);
        L0 += (diffuseColor + specularColor) * dirLight.color * LdotN;
    }
	vec3 ambient = vec3(0.001) * albedo;
    FragColor = vec4(L0 + ambient, 1);
}