#type vertex
#version 450 core
layout (location = 0) in vec3 aPos;

uniform mat4 projection;
uniform mat4 view;

out vec3 localPos;

void main()
{
    localPos = aPos;

    mat4 rotView = mat4(mat3(view)); // remove translation from the view matrix
    vec4 clipPos = projection * rotView * vec4(localPos, 1.0);

    gl_Position = clipPos;
}

#type fragment
#version 450 core
layout (location = 0) out vec4 FragColor;

in vec3 localPos;

layout (binding = 0) uniform samplerCube environmentMap;

void main()
{
    vec3 envColor = pow(texture(environmentMap, localPos).rgb, vec3(2.2));
    
    //envColor = envColor / (envColor + vec3(1.0));
    //envColor = pow(envColor, vec3(1.0/2.2)); 
    FragColor = vec4(envColor, 1.0);
}