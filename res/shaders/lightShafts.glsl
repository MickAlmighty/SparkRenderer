#type vertex
#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 textureCoords;

out vec2 texCoords;

void main()
{
    texCoords = textureCoords;
    gl_Position = vec4(position, 1);
}

#type fragment
#version 450
layout (location = 0) out vec4 FragColor;

in vec2 texCoords;

layout (binding = 0) uniform sampler2D depthTexture;
layout (binding = 1) uniform sampler2D lightingTexture;

uniform vec2 lightScreenPos;
uniform vec3 lightColor;

uniform float exposure = 0.0034f;
uniform float decay = 0.995f;
uniform float density = 0.95f;
uniform float weight = 6.65;

const int samples = 48;
const float oneStep = 1.0 / float(samples);
const float maxDistance = length(vec2(0.5f));

void main()
{
    vec4 outColor = vec4(vec3(0.0f), 1.0f);

    vec2 textureCoords = texCoords;
    vec2 texCoordStep = textureCoords - lightScreenPos;
    texCoordStep *= oneStep * density;

    float distanceToScreenCenter = smoothstep(0.0f, maxDistance, maxDistance - distance(vec2(0.5f), lightScreenPos));

    float illuminationDecay = 1.0f;

    for(int i = 0; i < samples; ++i)
    {
        // Step sample location along ray.
        textureCoords -= texCoordStep;

        // vec4 colorSample = vec4(lightColor, 1.0f);
        vec4 colorSample = texture(lightingTexture, clamp(textureCoords, 0.0f, 1.0f));

        float depth = texture(depthTexture, clamp(textureCoords, 0.0f, 1.0f)).x;
        if (depth > 0.00002f)
        {
            colorSample = vec4(0);
        }

        // Apply sample attenuation scale/decay factors.
        colorSample *= illuminationDecay * weight;

        // Accumulate combined color.  
        outColor += colorSample;

        // Update exponential decay factor.
        illuminationDecay *= decay;
    }

    FragColor = outColor * exposure * distanceToScreenCenter;
}