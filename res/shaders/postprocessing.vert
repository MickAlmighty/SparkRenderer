#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coords;

out vec2 tex_coords;

void main()
{
    tex_coords = texture_coords;
    gl_Position = vec4(position, 1);
}