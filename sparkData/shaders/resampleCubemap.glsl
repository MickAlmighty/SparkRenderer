#type vertex
#version 450
layout (location = 0) in vec3 aPos;

void main()
{
    gl_Position =  vec4(aPos, 1.0);
}

#type geometry
#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 18) out;

layout (location = 0) uniform mat4 projection;

layout(std430, binding = 0) readonly buffer Views
{
    mat4 views[6];
};

layout (location = 0) out vec3 cubemapCoord;

void main()
{
    for(int face = 0; face < 6; ++face)
    {
        for(int i = 0; i < 3; ++i)
        {
            cubemapCoord = gl_in[i].gl_Position.xyz;
            gl_Layer = face;
            gl_Position = projection * views[face] * gl_in[i].gl_Position;
            EmitVertex();
        }
        EndPrimitive();
    }
}

#type fragment
#version 450
layout (location = 0) out vec4 FragColor;

layout (binding = 0) uniform samplerCube inputCubemap;

layout (location = 0) in vec3 cubemapCoord;

void main()
{
    FragColor = texture(inputCubemap, cubemapCoord);
}