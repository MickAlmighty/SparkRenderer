#type vertex
#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coords;
layout (location = 0) out vec2 tex_coords;

void main()
{
    tex_coords = texture_coords;
    gl_Position = vec4(position, 1);
}

#type fragment
#version 450
layout (location = 0) in vec2 tex_coords;
layout (location = 0) out vec4 FragColor;

layout (binding = 0) uniform sampler2D inputTexture;

void main()
{
    FragColor = texture(inputTexture, tex_coords);
}