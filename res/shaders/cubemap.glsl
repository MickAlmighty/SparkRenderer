#type vertex
#version 450 core
layout (location = 0) in vec3 aPos;

layout (std140) uniform Camera
{
    vec4 pos;
    mat4 view;
    mat4 projection;
    mat4 invertedView;
    mat4 invertedProjection;
} camera;

out vec3 texCoords;

void main()
{
    texCoords = aPos;

    mat4 rotView = mat4(mat3(camera.view)); // remove translation from the view matrix
    vec4 clipPos = camera.projection * rotView * vec4(aPos, 0.0);
    clipPos.z = 0; // assigning 0 for inversed Z depth buffer (0 is on a far plane)
    gl_Position = clipPos;
    // if farPlane depth is 1 then you need to 
    //gl_Position = clipPos.xyww;
}

#type fragment
#version 450 core
layout (location = 0) out vec4 FragColor;

in vec3 texCoords;

layout (binding = 0) uniform samplerCube environmentMap;

vec3 attenuateHighFrequencies(vec3 color)
{
    const float luma = dot(color, vec3(0.299, 0.587, 0.114));
    float weight = 1 / (1 + luma * 0.1f);
    return color * weight;
}

void main()
{
    //vec3 envColor = pow(texture(environmentMap, texCoords).rgb, vec3(2.2));
    vec3 envColor = texture(environmentMap, texCoords).rgb;
    FragColor = vec4(attenuateHighFrequencies(envColor), 1.0);
}