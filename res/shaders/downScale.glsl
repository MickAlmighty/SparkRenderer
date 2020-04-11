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
uniform bool downscale = true;
uniform float intensity = 1.0f;
uniform vec2 outputTextureSizeInversion; //equals to 1.0f / texture size 

in vec2 texCoords;

vec2 texelOffset(vec2 offset)
{
    return offset * outputTextureSizeInversion;
    //return vec2(float(offset.x) * outputTextureSizeInversion.x, float(offset.y) * outputTextureSizeInversion.y);
}

#define sampleInputRGB(x) texture(inputTexture, x).xyz

vec3 sampleTexture()
{
    vec3 outputColor = vec3(0);

    vec3 center = sampleInputRGB(texCoords);

    {// center quad
        vec3 bottomLeft = sampleInputRGB(texCoords + texelOffset(vec2(-0.5f)));
        vec3 bottomRight = sampleInputRGB(texCoords + texelOffset(vec2(0.5f, -0.5f)));
        vec3 topRight = sampleInputRGB(texCoords + texelOffset(vec2(0.5f)));
        vec3 topLeft = sampleInputRGB(texCoords + texelOffset(vec2(-0.5f, 0.5f)));
        
        vec3 centerQuadAverage = 0.25 * (bottomLeft + bottomRight + topRight + topLeft);
        outputColor += centerQuadAverage * 0.5f; // 0.5f is a total weight
    }

    vec3 outerMiddleLeft = sampleInputRGB(texCoords + texelOffset(vec2(-1.0f, 0.0f)));
    vec3 outerMiddleRight = sampleInputRGB(texCoords + texelOffset(vec2(1.0f, 0.0f)));
    vec3 outerTopMiddle = sampleInputRGB(texCoords + texelOffset(vec2(0.0f, 1.0f)));
    vec3 outerBottomMiddle = sampleInputRGB(texCoords + texelOffset(vec2(0.0f, -1.0f)));

    vec3 outerTopLeft = sampleInputRGB(texCoords + texelOffset(vec2(-1.0f, 1.0f)));
    vec3 outerTopRight = sampleInputRGB(texCoords + texelOffset(vec2(1.0f, 1.0f)));
    vec3 outerBottomRight = sampleInputRGB(texCoords + texelOffset(vec2(1.0f, -1.0f)));
    vec3 outerBottomLeft = sampleInputRGB(texCoords + texelOffset(vec2(0.0f, 0.0f)));

    {//top left quad
        vec3 topLeftQuadAverage = 0.25f * (center + outerTopMiddle + outerTopLeft + outerMiddleLeft);
        outputColor += 0.125f * topLeftQuadAverage;
    }

    {//top right quad
        vec3 topRightQuadAverage = 0.25f * (center + outerTopMiddle + outerTopRight + outerMiddleRight);
        outputColor += 0.125f * topRightQuadAverage;
    }

    {//bottom right quad
        vec3 bottomRightQuadAverage = 0.25f * (center + outerBottomMiddle + outerBottomRight + outerMiddleRight);
        outputColor += 0.125f * bottomRightQuadAverage;
    }

    {//bottom right quad
        vec3 bottomLeftQuadAverage = 0.25f * (center + outerBottomMiddle + outerBottomLeft + outerMiddleLeft);
        outputColor += 0.125f * bottomLeftQuadAverage;
    }
    return outputColor;
}

void main()
{
    if (downscale)
        FragColor = vec4(sampleTexture() * max(intensity, 0.0f), 0.0f); // 13 samples
    else
        FragColor = vec4(sampleInputRGB(texCoords) * max(intensity, 0.0f), 0.0f); // one center sample
}