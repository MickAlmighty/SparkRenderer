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

uniform vec2 outputTextureSizeInversion; //equals to 1.0f / texture size 

in vec2 texCoords;

vec2 texelOffset(vec2 offset)
{
    return offset * outputTextureSizeInversion;
}

#define sampleInputRGB(x) texture(inputTexture, x).xyz

vec3 sampleTexture()
{
    vec3 outputColor = vec3(0);
    vec3 sample1 = sampleInputRGB(texCoords + texelOffset(vec2(-1.0f, 1.0f)));
    vec3 sample2 = sampleInputRGB(texCoords + texelOffset(vec2(0.0f, 1.0f)));
    vec3 sample3 = sampleInputRGB(texCoords + texelOffset(vec2(1.0f, 1.0f)));
    vec3 sample4 = sampleInputRGB(texCoords + texelOffset(vec2(-0.5f, 0.5f)));
    vec3 sample5 = sampleInputRGB(texCoords + texelOffset(vec2(0.5f, 0.5f)));
    vec3 sample6 = sampleInputRGB(texCoords + texelOffset(vec2(-1.0f, 0.0f)));
    vec3 sample7 = sampleInputRGB(texCoords);
    vec3 sample8 = sampleInputRGB(texCoords + texelOffset(vec2(1.0f, 0.0f)));
    vec3 sample9 = sampleInputRGB(texCoords + texelOffset(vec2(-0.5f, -0.5f)));
    vec3 sample10 = sampleInputRGB(texCoords + texelOffset(vec2(0.5f, -0.5f)));
    vec3 sample11 = sampleInputRGB(texCoords + texelOffset(vec2(-1.0f, -1.0f)));
    vec3 sample12 = sampleInputRGB(texCoords + texelOffset(vec2(0.0f, -1.0f)));
    vec3 sample13 = sampleInputRGB(texCoords + texelOffset(vec2(0.0f, -1.0f)));

    vec3 redSamples = ((sample4 + sample5 + sample9 + sample10) * 0.25f) * 0.5f;
    vec3 yellowSamples = ((sample1 + sample2 + sample6 + sample7) * 0.25f) * 0.125f;
    vec3 greenSamples = ((sample2 + sample3 + sample7 + sample8) * 0.25f) * 0.125f;
    vec3 blueSamples = ((sample7 + sample8 + sample12 + sample13) * 0.25f) * 0.125f;
    vec3 purpleSamples = ((sample6 + sample7 + sample11 + sample12) * 0.25f) * 0.125f;
    outputColor = redSamples + yellowSamples + greenSamples + blueSamples + purpleSamples;
    return outputColor;
}

void main()
{
    FragColor = vec4(sampleTexture(), 0.0f); // 13 samples
}