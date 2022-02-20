#type vertex
#version 450
layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 textureCoords;

layout (location = 0) out vec2 texCoords;

void main()
{
    texCoords = textureCoords;
    gl_Position = vec4(pos, 1.0);
}

#type fragment
#version 450
layout (location = 0) out vec4 FragColor;

layout (binding = 0) uniform sampler2D inputTexture;

layout (location = 0) in vec2 texCoords;

void main()
{
    FragColor = texture(inputTexture, texCoords);
}