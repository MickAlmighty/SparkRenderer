#type vertex
#version 450
layout (location = 0) in vec3 position;

out vec3 cubemapCoords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    cubemapCoords = position;  
    gl_Position =  projection * view * vec4(position, 1.0);
}

#type fragment
#version 450
layout (location = 0) out vec4 FragColor;

layout (binding = 0) uniform samplerCube inputCubemap;

in vec3 cubemapCoords;

void main()
{
    FragColor = texture(inputCubemap, cubemapCoords);
}