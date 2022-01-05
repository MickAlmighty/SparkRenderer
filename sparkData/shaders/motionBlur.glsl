#type vertex
#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coords;

out vec2 texCoords;

void main()
{
    texCoords = texture_coords;
    gl_Position = vec4(position, 1.0);
}

#type fragment
#version 450

layout (location = 0) out vec3 FragColor;

noperspective in vec2 texCoords;

layout (binding = 0) uniform sampler2D colorTexture;
layout (binding = 1) uniform sampler2D depthTexture;

uniform mat4 prevViewProj;
uniform float blurScale = 1.0f;
uniform vec2 texelSize;

layout (std140) uniform Camera
{
    vec4 pos;
    mat4 view;
    mat4 projection;
    mat4 invertedView;
    mat4 invertedProjection;
} camera;

vec4 worldPosFromDepth(float depth, mat4 invProj, mat4 invView) {
    float z = depth;
    vec4 ndcPosition = vec4(texCoords * 2.0 - 1.0, z, 1.0);
    vec4 viewSpacePosition = invProj * ndcPosition;

    // inverse of perspective division
    viewSpacePosition /= viewSpacePosition.w;

    vec4 worldSpacePosition = invView * viewSpacePosition;

    return worldSpacePosition;
}

#define MAX_SAMPLES 12
#define ONE_BY_MAX_SAMPLES 1.0 / float(MAX_SAMPLES)

void main()
{
    const float depthValue = texture(depthTexture, texCoords).x;

    const vec4 worldPos = worldPosFromDepth(depthValue, camera.invertedProjection, camera.invertedView);
    const vec4 previousClipSpacePos = prevViewProj * worldPos;
    const vec2 previousNdcPos = (previousClipSpacePos.xyz / previousClipSpacePos.w).xy;

    //screen space vectors
    const vec2 currentPos = texCoords;
    const vec2 previousPos = previousNdcPos.xy * 0.5 + 0.5;

    const vec2 velocityNotScaled = previousPos - currentPos;
    const vec2 velocity = velocityNotScaled * blurScale;

    const vec2 blurStartTexCoords = texCoords - velocity * 0.5f;

    vec3 color = vec3(0);
    for (uint i = 0; i < MAX_SAMPLES; ++i)
    {
        color += texture(colorTexture, blurStartTexCoords + velocity * (float(i) * ONE_BY_MAX_SAMPLES)).rgb;
    }

    FragColor = color * ONE_BY_MAX_SAMPLES;
}