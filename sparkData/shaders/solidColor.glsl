#type vertex
#version 450
#include "Camera.hglsl"
layout (location = 0) in vec3 Position;

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
    vec4 worldPosition = u_Uniforms.model * vec4(Position, 1);
    gl_Position = camera.projection * camera.view * worldPosition;
}

#type fragment
#version 450
layout (location = 0) out vec3 FragColor;

layout (push_constant) uniform Color
{
    vec3 color;
} u_Uniforms2;

void main() 
{
    FragColor = u_Uniforms2.color;
}