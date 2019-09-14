#version 450
layout (location = 0) out vec3 Position;
layout (location = 1) out vec4 FragColor;
layout (location = 2) out vec3 Normal;

layout (binding = 1) uniform sampler2D diffuseTexture;
layout (binding = 2) uniform sampler2D normalTexture;

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
}