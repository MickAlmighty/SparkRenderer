#version 450
layout (location = 0) out vec4 FragColor;

layout (binding = 0) uniform sampler2D screenTexture;

in vec2 tex_coords;

void main()
{
    FragColor = texture(screenTexture, tex_coords);
}