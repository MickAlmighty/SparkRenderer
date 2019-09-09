#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texture_coords;

uniform mat4 MVP;

out vec2 tex_coords;

void main()
{
    gl_Position = MVP * vec4(position, 1);
    tex_coords = texture_coords;
}