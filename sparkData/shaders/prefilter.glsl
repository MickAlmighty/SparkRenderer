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
#version 450 core
layout (location = 0) out vec4 FragColor;
in vec3 cubemapCoord;

layout (binding = 0) uniform samplerCube environmentMap;
uniform float roughness;
uniform float textureSize;

const float PI = 3.14159265359;
const uint SAMPLE_COUNT = 32u;

float RadicalInverse_VdC(uint bits);
vec2 Hammersley(uint i, uint N);
vec3 ImportanceSampleGGX(vec2 Xi, float roughness, mat3 tangentToWorld);
float DistributionGGX(float NdotH, float roughness);

void main()
{
    vec3 N = normalize(cubemapCoord);    
    vec3 V = N;

    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    mat3 tangentToWorld = mat3(tangent, bitangent, N);

    float totalWeight = 0.0;   
    vec3 prefilteredColor = vec3(0.0);
    for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H  = ImportanceSampleGGX(Xi, roughness, tangentToWorld);
        float NdotH = max(dot(N, H), 0.0);
        vec3 L  = normalize(2.0 * NdotH * H - V);

        float NdotL = max(dot(N, L), 0.0);
        if(NdotL > 0.0)
        {
            // sample from the environment's mip level based on roughness/pdf
            float D = DistributionGGX(NdotH, roughness);
            float pdf = D * NdotH / (4.0 * NdotH) + 0.0001; 

            float resolution = textureSize; // resolution of source cubemap (per face)
            float saTexel = 4.0 * PI / (6.0 * resolution * resolution);
            float saSample = 1.0 / (float(SAMPLE_COUNT) * pdf + 0.0001);

            float mipLevel = roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel);

            prefilteredColor += textureLod(environmentMap, L, mipLevel).rgb * NdotL;
            totalWeight += NdotL;
        }
    }
    prefilteredColor = prefilteredColor / totalWeight;

    FragColor = vec4(prefilteredColor, 1.0);
}  

float RadicalInverse_VdC(uint bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 Hammersley(uint i, uint N)
{
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
} 

vec3 ImportanceSampleGGX(vec2 Xi, float roughness, mat3 tangentToWorld)
{
    float a = roughness*roughness;

    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);

    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    return normalize(tangentToWorld * H);
} 

float DistributionGGX(float NdotH, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}