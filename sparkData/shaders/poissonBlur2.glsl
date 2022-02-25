#type vertex
#version 450
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 textureCoords;

layout (location = 0) out vec2 texCoords;

void main() 
{
    texCoords = textureCoords;
    gl_Position = vec4(pos, 1.0);
}

#type fragment
#version 450
layout (location = 0) in vec2 texCoords;
layout (location = 0) out vec4 FragColor;

layout (binding = 0) uniform sampler2D image;

layout (push_constant) uniform PushConstants
{
    float scale;
} u_Uniforms;

layout (binding = 0) buffer Taps
{
    vec2 taps[16];
};

#define TAPS 16

void main()
{
    const vec4 color = texture(image, texCoords);
    const float CoC = color.a;
    vec3 maxVal = color.rgb;

    const vec2 texSize = vec2(textureSize(image, 0).xy);
    const vec2 screenRatioScaling = vec2(texSize.y / texSize.x, 1.0f);

    for (int i = 0; i < TAPS - 1; i++)
    {
        const vec2 offset = taps[i] * screenRatioScaling * u_Uniforms.scale * CoC;
        const vec3 tap = texture(image, texCoords + offset).rgb;
        maxVal = max(tap, maxVal);
    }

    FragColor =  vec4(maxVal, CoC);
}