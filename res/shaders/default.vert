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