#version 450
layout (location = 0) out vec4 FragColor;

layout (binding = 1) uniform sampler2D diffuse;

in vec2 tex_coords;

void main()
{
    FragColor = texture(diffuse, tex_coords);
}