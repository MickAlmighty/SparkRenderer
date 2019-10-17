#type vertex
#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texture_coords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 tex_coords;

out TO_TEXTURES {
	vec3 position;
	vec3 normal;
} to_textures;

void main()
{
	mat3 normalMatrix = transpose(inverse(mat3(view * model)));
	vec4 worldPosition = model * vec4(position, 1);
	to_textures.normal = normalMatrix * normal;

	to_textures.position = vec3(worldPosition.xyz);
    gl_Position = projection * view * worldPosition;
    tex_coords = texture_coords;
}

#type fragment
#version 450
layout (location = 0) out vec3 Position;
layout (location = 1) out vec4 FragColor;
layout (location = 2) out vec3 Normal;
layout (location = 3) out float Roughness;
layout (location = 4) out float Metalness;

layout (binding = 1) uniform sampler2D diffuseTexture;
layout (binding = 2) uniform sampler2D normalTexture;
layout (binding = 3) uniform sampler2D roughnessTexture;
layout (binding = 4) uniform sampler2D metalnessTexture;

in vec2 tex_coords;

in TO_TEXTURES {
	vec3 position;
	vec3 normal;
} to_textures;

void main()
{
	Position = to_textures.position;
    FragColor = texture(diffuseTexture, tex_coords);
    //Normal = to_textures.normal;
    Normal = texture(normalTexture, tex_coords).xyz;
	Roughness = texture(roughnessTexture, tex_coords).x;
	Metalness = texture(metalnessTexture, tex_coords).x;
}