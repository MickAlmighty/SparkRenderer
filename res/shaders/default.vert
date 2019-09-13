#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texture_coords;

uniform mat4 MVP;
uniform mat4 View;
uniform mat4 Model;

out vec2 tex_coords;

out TO_TEXTURES {
	vec3 position;
	vec3 normal;
} to_textures;

void main()
{
	mat3 normalMatrix = transpose(inverse(mat3(View * Model)));
	to_textures.normal = normalMatrix * normal;
	to_textures.position = vec3(Model * vec4(position, 1));
    gl_Position = MVP * vec4(position, 1);
    tex_coords = texture_coords;
}