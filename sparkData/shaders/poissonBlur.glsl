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
layout (binding = 1) uniform sampler2D depth;

layout (push_constant) uniform PushConstants
{
    float scale;
} u_Uniforms;

#define TAPS 16
#define TAPS_INV 1.0 / float(TAPS)

layout (binding = 0) buffer Taps
{
    vec2 taps[TAPS];
};

void main()
{
    const vec2 texSize = vec2(textureSize(image, 0).xy);
    const vec2 screenRatioScaling = vec2(texSize.y / texSize.x, 1.0f);

    vec4 color = texture(image, texCoords);
    const float pxDepth = texture(depth, texCoords).x;
    const float CoC = color.a;
    float totalWeight = 0.0f;
    for (int i = 0; i < TAPS - 1; i++)
    {
        const vec2 offset = taps[i] * screenRatioScaling * u_Uniforms.scale * CoC;
        const vec2 coords = texCoords + offset;
        const vec4 samplePixel = texture(image, coords);
        const float sampleDepth = texture(depth, coords).x;

        const float bleedingBias = 0.02f;
        const float bleedingMult = 30.0f;
        float weight = sampleDepth > pxDepth ? samplePixel.a * bleedingMult : 1.0f;
        weight = (CoC > (samplePixel.a + bleedingBias)) ? weight : 1.0f;
        weight = clamp(weight, 0.0f, 1.0f);

        color += samplePixel * weight;
        totalWeight += weight;
    }

    FragColor = color / totalWeight;
}