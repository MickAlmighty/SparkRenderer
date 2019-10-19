#type vertex
#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texture_coords;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 tex_coords;
out mat3 TBN_matrix;

out TO_TEXTURES {
	vec3 position;
	vec3 normal;
} to_textures;

void main()
{
	vec4 worldPosition = model * vec4(position, 1);

	mat3 normalMatrix = transpose(inverse(mat3(model)));
	vec3 T = normalize(normalMatrix * tangent);
	vec3 N = normalize(normalMatrix * normal);
	//T = normalize(T - dot(T, N) * N);
	//vec3 B = cross(N, T);
	vec3 B = normalize(normalMatrix * bitangent);
	mat3 TBN = mat3(T, B, N);
	TBN_matrix = TBN;

	to_textures.normal = normalMatrix * normal;
	to_textures.position = worldPosition.xyz;
   
	tex_coords = texture_coords;

	gl_Position = projection * view * worldPosition;
}

#type fragment
#version 450
layout (location = 0) out vec3 Position;
layout (location = 1) out vec4 FragColor;
layout (location = 2) out vec3 ModelNormal;
layout (location = 3) out vec3 Normal;
layout (location = 4) out float Roughness;
layout (location = 5) out float Metalness;

layout (binding = 1) uniform sampler2D diffuseTexture;
layout (binding = 2) uniform sampler2D normalTexture;
layout (binding = 3) uniform sampler2D roughnessTexture;
layout (binding = 4) uniform sampler2D metalnessTexture;

in vec2 tex_coords;
in mat3 TBN_matrix;

in TO_TEXTURES {
	vec3 position;
	vec3 normal;
} to_textures;

void main()
{
	Position = to_textures.position;
    FragColor = texture(diffuseTexture, tex_coords);
	ModelNormal = normalize(to_textures.normal);

	vec3 normalFromTexture = texture(normalTexture, tex_coords).xyz;
	normalFromTexture = normalize(normalFromTexture * 2.0 - 1.0);
	Normal = normalize(TBN_matrix * normalFromTexture);
	Roughness = texture(roughnessTexture, tex_coords).x;
	Metalness = texture(metalnessTexture, tex_coords).x;
}