#type vertex
#version 450
layout (location = 0) in vec3 Position;
layout (location = 1) in vec2 TextureCoords;

layout (location = 0) out vec2 texCoords;

void main()
{
    texCoords = TextureCoords;
    gl_Position = vec4(Position, 1.0f);
}

#type fragment
#version 450
#include "Camera.hglsl"
layout (location = 0) in vec2 texCoords;
layout (location = 0) out float AmbientOcclusion;

layout (binding = 0) uniform sampler2D depthTexture;
layout (binding = 1) uniform sampler2D normalTexture;
layout (binding = 2) uniform sampler2D texNoise;

layout (std140, binding = 0) uniform Camera
{
    CameraData camera;
};

layout (std140, binding = 1) uniform Samples
{
    vec4 samples[64];
};

layout (push_constant) uniform Uniforms
{
    float radius;
    float bias;
    float power;
    int kernelSize;
    vec2 screenSize;
} u_Uniforms;

vec4 viewSpacePosFromDepth(float depth, mat4 invProj, vec2 uv)
{
    vec4 clipSpacePosition = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewSpacePosition = invProj * clipSpacePosition;

    // Perspective division
    viewSpacePosition /= viewSpacePosition.w;

    return viewSpacePosition;
}

vec3 decode(vec2 enc)
{
    vec2 fenc = enc * 4.0f - 2.0f;
    float f = dot(fenc, fenc);
    float g = sqrt(1.0f - f / 4.0f);
    vec3 n;
    n.xy = fenc*g;
    n.z = 1.0f - f / 2.0f;
    return n;
}

vec3 getViewSpacePosition(vec2 uv)
{
    float depth = texture(depthTexture, uv).x;
    return viewSpacePosFromDepth(depth, camera.invertedProjection, uv).xyz;
}

vec3 getNormal(vec2 uv)
{
    return normalize(decode(texture(normalTexture, uv).xy));
}

void main() 
{
    float depth = texture(depthTexture, texCoords).x;
    if (depth == 0.0f)
    {
        discard;
    }

    const vec2 noiseScale = vec2(u_Uniforms.screenSize.x / 4.0f, u_Uniforms.screenSize.y / 4.0f);

    vec3 fragPos = viewSpacePosFromDepth(depth, camera.invertedProjection, texCoords).xyz;
    vec3 N = getNormal(texCoords);
    vec3 randomVec = normalize(texture(texNoise, texCoords * noiseScale).xyz) * 2.0 - 1.0;

    // create TBN change-of-basis matrix: from tangent-space to view-space
    vec3 T = normalize(randomVec - N * dot(randomVec, N));
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);

    float occlusion = 0.0f;
    for(int i = 0; i < u_Uniforms.kernelSize; ++i)
    {
        vec3 sampleP = TBN * samples[i].xyz; // from tangent to view-space
        sampleP = fragPos + sampleP * u_Uniforms.radius;

        vec4 offset = vec4(sampleP, 1.0);
        offset = camera.projection * offset;    // from view to clip-space
        offset.xy /= offset.w;               // perspective divide
        offset.xy = offset.xy * 0.5 + 0.5; // transform to range 0.0 - 1.0 

        float sampleDepth = getViewSpacePosition(offset.xy).z;

        float rangeCheck = smoothstep(0.0f, 1.0f, u_Uniforms.radius / abs(fragPos.z - sampleDepth));
        occlusion += (sampleDepth >= sampleP.z + u_Uniforms.bias ? 1.0 : 0.0) * rangeCheck;
    }

    occlusion = 1.0f - (occlusion / u_Uniforms.kernelSize);
    AmbientOcclusion.x = clamp(pow(occlusion, u_Uniforms.power), 0.0f, 1.0f);
}