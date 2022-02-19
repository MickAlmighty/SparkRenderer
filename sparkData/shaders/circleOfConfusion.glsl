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
layout (location = 0) out float CircleOfConfusion;

layout (binding = 0) uniform sampler2D depthTexture;

layout (std140, binding = 0) uniform Camera
{
    CameraData camera;
};

//depth range for reversed Z depth buffer
layout (push_constant) uniform PushConstants
{
    float zNear;
    float zNearEnd;
    float zFarStart ;
    float zFar;
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

float getCircleOfConfusion(float pixelDepth, float nearDepth, float farDepth)
{
    // -1.0f -> camera points to negative z values 
    return clamp((-1.0f * pixelDepth - nearDepth) / (farDepth - nearDepth), 0.0f, 1.0f);
}

void main()
{
    vec3 viewPos = getViewSpacePosition(texCoords);
    float nearCoc = 1.0f - getCircleOfConfusion(viewPos.z, u_Uniforms.zNear, u_Uniforms.zNearEnd);
    float farCoc = getCircleOfConfusion(viewPos.z, u_Uniforms.zFarStart, u_Uniforms.zFar) * 0.4f;
    CircleOfConfusion.x = max(nearCoc, farCoc);
}