#type vertex
#version 450
layout (location = 0) in vec3 pos;

uniform mat4 VP;

void main()
{
    gl_Position = VP * vec4(pos, 1.0);
}

#type fragment
#version 450
layout (location = 0) out vec4 FragPos;

void main()
{
    FragPos = vec4(0.0, 1.0, 0.0, 1.0);
}