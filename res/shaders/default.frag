#version 450
layout (location = 0) out vec4 FragColor;

in vec2 tex_coords;

void main()
{
    FragColor = vec4(1, 0, 1, 1);
}