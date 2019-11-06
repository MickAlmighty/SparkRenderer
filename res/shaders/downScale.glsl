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
layout (binding = 1) uniform sampler2D toBlendTexture;

uniform bool blend = false;

in vec2 texCoords;

void main()
{
    if (blend)
    {
        vec4 inputColor = texture(inputTexture, texCoords);
        vec4 colorToBlend = texture(toBlendTexture, texCoords);
        //vec4 color = mix(inputColor, colorToBlend, 0.5);
        vec4 color = inputColor + colorToBlend;
        FragColor = color;
    }
    else 
    {
        FragColor = texture(inputTexture, texCoords);
    }
}