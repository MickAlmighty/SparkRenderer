#type vertex
#version 450
#include "Camera.hglsl"
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texture_coords;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;

layout (push_constant) uniform Model
{
    mat4 model;
} u_Uniforms;

layout (std140, binding = 0) uniform Camera
{
    CameraData camera;
};

void main()
{
    vec4 worldPosition = u_Uniforms.model * vec4(position, 1);
    gl_Position = camera.viewProjection * worldPosition;
}