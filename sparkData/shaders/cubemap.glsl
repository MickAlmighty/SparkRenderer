#type vertex
#version 450
#include "Camera.hglsl"
layout (location = 0) in vec3 aPos;

layout (std140, binding = 0) uniform Camera
{
    CameraData camera;
};

layout (location = 0) out vec3 texCoords;

void main()
{
    texCoords = aPos;

    mat4 rotView = mat4(mat3(camera.view)); // remove translation from the view matrix
    vec4 clipPos = camera.projection * rotView * vec4(aPos, 0.0);
    clipPos.z = 0.0; // assigning 0 for inversed Z depth buffer (0 is on a far plane)
    gl_Position = clipPos;
    // if farPlane depth is 1 then you need to 
    //gl_Position = clipPos.xyww;
}

#type fragment
#version 450
layout (location = 0) out vec4 FragColor;

layout (location = 0) in vec3 texCoords;

layout (binding = 0) uniform samplerCube environmentMap;

void main()
{
    //vec3 envColor = pow(texture(environmentMap, texCoords).rgb, vec3(2.2));
    vec3 envColor = texture(environmentMap, texCoords).rgb;
    FragColor = vec4(envColor, 1.0);
}