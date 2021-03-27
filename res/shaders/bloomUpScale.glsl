#type vertex
#version 450
layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 textureCoords;

out vec2 texCoords;

void main()
{
    texCoords = textureCoords;
    gl_Position = vec4(pos, 1.0);
}

#type fragment
#version 450
layout (location = 0) out vec4 FragColor;

layout (binding = 0) uniform sampler2D inputTexture;

uniform float intensity = 1.0f;

in vec2 texCoords;

void main()
{
    FragColor = vec4(texture(inputTexture, texCoords).rgb * max(intensity, 0.0f), 0.0f); // one center sample
}