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
uniform float threshold = 0.5f;
uniform float thresholdSize = 1.0f;

in vec2 texCoords;

vec2 texelOffset(vec2 offset)
{
    return offset * outputTextureSizeInversion;
}

#define sampleInputRGB(x) texture(inputTexture, x).xyz

float luma(vec3 color)
{
    return dot(color, vec3(0.299, 0.587, 0.114));
}

vec3 averageColor(vec3 color)
{
    return color * (1.0f / (1.0f + luma(color)));
}


vec3 getBringhtPassColor(vec3 color, float threshold_, float thresholdSize_)
{
    threshold_ = max(threshold_, 0.0f);
    thresholdSize_ = max(threshold_ + 0.1f, thresholdSize_);
    return smoothstep(threshold_, threshold_ + thresholdSize_, luma(color)) * color;
}

vec3 sampleTexture()
{
    
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

    vec3 redSamplesAvg = (sample4 + sample5 + sample9 + sample10) * 0.25f;
    vec3 redHBox = averageColor(redSamplesAvg) * 0.5f;

    vec3 yellowSamplesAvg = (sample1 + sample2 + sample6 + sample7) * 0.25f;
    vec3 yellowHBox = averageColor(yellowSamplesAvg) * 0.125f;

    vec3 greenSamplesAvg = (sample2 + sample3 + sample7 + sample8) * 0.25f;
    vec3 greenHBox = averageColor(greenSamplesAvg) * 0.125f;

    vec3 blueSamplesAvg = (sample7 + sample8 + sample12 + sample13) * 0.25f;
    vec3 blueHBox = averageColor(blueSamplesAvg) * 0.125f;

    vec3 purpleSamplesAvg = (sample6 + sample7 + sample11 + sample12) * 0.25f;
    vec3 purpleHBox = averageColor(purpleSamplesAvg) * 0.125f;

    vec3 outputColor = redHBox + yellowHBox + greenHBox + blueHBox + purpleHBox;

    return getBringhtPassColor(outputColor, threshold, thresholdSize);
}

void main()
{
    FragColor = vec4(sampleTexture(), 0.0f); // 13 samples
}