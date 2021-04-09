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

uniform mat4 projection;

layout(std430) readonly buffer Views
{
    mat4 views[]; // 6 matrices
};

out vec3 cubemapCoord;

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
in vec3 cubemapCoord;

layout (binding = 0) uniform sampler2D equirectangularMap;

const vec2 invAtan = vec2(0.1591, 0.3183);
vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= -invAtan;
    uv += 0.5;
    return uv;
}

vec3 attenuateHighFrequencies(vec3 color)
{
    const float luma = dot(color, vec3(0.299, 0.587, 0.114));
    float weight = 1 / (1 + luma * 0.05f);
    return color * weight;
}

void main()
{
    vec2 uv = SampleSphericalMap(normalize(cubemapCoord));
    vec3 color = texture(equirectangularMap, uv).rgb;

    FragColor = vec4(attenuateHighFrequencies(color), 1.0);
}