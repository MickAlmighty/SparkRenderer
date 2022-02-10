#type vertex
#version 450
layout (location = 0) in vec3 Position;

layout (location = 0) uniform mat4 model;

layout (std140, binding = 0) uniform Camera
{
    vec4 pos;
    mat4 view;
    mat4 projection;
    mat4 invertedView;
    mat4 invertedProjection;
} camera;

void main()
{
    vec4 worldPosition = model * vec4(Position, 1);
    gl_Position = camera.projection * camera.view * worldPosition;
}

#type fragment
#version 450
layout (location = 0) out vec3 FragColor;

layout (location = 1) uniform vec3 color = vec3(1.0f, 1.0f, 1.0f);

void main() 
{
    FragColor = color;
}