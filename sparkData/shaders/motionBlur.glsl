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

#define MAX_SAMPLES 32

vec4 worldPosFromDepth(float depth, mat4 invProj, mat4 invView) {
    float z = depth;
    // if (z == 0.0f)
    // {
    //     z = 0.01f;
    // }
    vec4 clipSpacePosition = vec4(texCoords * 2.0 - 1.0, z, 1.0);
    vec4 viewSpacePosition = invProj * clipSpacePosition;

    // Perspective division
    viewSpacePosition /= viewSpacePosition.w;

    vec4 worldSpacePosition = invView * viewSpacePosition;

    return worldSpacePosition;
}

vec4 viewPosFromDepth(float depth, mat4 invProj) {
    float z = depth;
    if (z == 0.0f)
    {
        z = 0.05f;
    }
    vec4 clipSpacePosition = vec4(texCoords * 2.0 - 1.0, z, 1.0);
    vec4 viewSpacePosition = invProj * clipSpacePosition;

    // Perspective division
    viewSpacePosition /= viewSpacePosition.w;

    return viewSpacePosition;
}

void main()
{
    float depthValue = texture(depthTexture, texCoords).x;
    vec3 color = texture(colorTexture, texCoords).rgb;
    
    vec4 worldPos = worldPosFromDepth(depthValue, camera.invertedProjection, camera.invertedView);
    vec4 previousViewPos = prevViewProj * worldPos;
    previousViewPos.xyz /= previousViewPos.w;

    //screen space vectors
    vec2 currentPos = texCoords;
    vec2 previousPos = previousViewPos.xy * 0.5 + 0.5;
    
    vec2 velocity = (previousPos.xy - currentPos.xy);
    velocity *= blurScale;
    
    float speed = length(velocity / texelSize);
    
    int numSamples = clamp(int(speed), 1, MAX_SAMPLES);
    for ( uint i = 1; i < numSamples; ++i)
    {
        vec2 offset = velocity * (float(i) / float(numSamples - 1) - 0.5);
        vec3 currentColor = texture(colorTexture, texCoords + offset).rgb;
        color += currentColor;
    }
    //FragColor = vec3(velocity.xyy);
    FragColor = color / float(numSamples);
}