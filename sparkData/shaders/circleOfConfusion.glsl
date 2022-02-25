#type vertex
#version 450
layout (location = 0) in vec3 Position;
layout (location = 1) in vec2 TextureCoords;

layout (location = 0) out vec2 texCoords;

void main()
{
    texCoords = TextureCoords;
    gl_Position = vec4(Position, 1.0f);
}

#type fragment
#version 450
#include "Camera.hglsl"
layout (location = 0) in vec2 texCoords;
layout (location = 0) out vec4 FragColor;

layout (binding = 0) uniform sampler2D depthTexture;
layout (binding = 1) uniform sampler2D lightingTexture;

layout (std140, binding = 0) uniform Camera
{
    CameraData camera;
};

//depth range for reversed Z depth buffer
layout (push_constant) uniform PushConstants
{
    float A;        //aperture
    float f;        //focal length
    float S1;       //focal distance
    float maxCoC;   //max CoC diameter
} u_Uniforms;

vec4 viewPosFromDepth(float depth, mat4 invProj, vec2 uv) 
{
    vec4 clipSpacePosition = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewSpacePosition = invProj * clipSpacePosition;

    // Perspective division
    viewSpacePosition /= viewSpacePosition.w;

    return viewSpacePosition;
}

vec3 getViewSpacePosition(vec2 uv)
{
    float depth = texture(depthTexture, uv).x;
    return viewPosFromDepth(depth, camera.invertedProjection, uv).xyz;
}

void main()
{
    vec3 viewPos = getViewSpacePosition(texCoords);
    const vec3 color = texture(lightingTexture, texCoords).xyz;

    const float S2 = -viewPos.z; // make distance from camera positive by negation
    const float denominator = (u_Uniforms.S1 - u_Uniforms.f) * S2;
    const float nominator = u_Uniforms.A * u_Uniforms.f * abs(S2 - u_Uniforms.S1);
    const float CoC = nominator / denominator;
    const float sensorHeight = 0.024f;
    const float percentOfSensor = CoC / sensorHeight;
    const float blurFactor = clamp(percentOfSensor, 0, u_Uniforms.maxCoC);
    FragColor = vec4(color, blurFactor);
}