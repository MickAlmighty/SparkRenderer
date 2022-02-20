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
layout (location = 0) in vec2 texCoords;
layout (location = 0) out float AmbientOcclusion;

layout (push_constant) uniform PushConstants
{
    int blurSize;
} u_Uniforms;

layout (binding = 0) uniform sampler2D uTexInput;

void main() 
{
    vec2 texelSize = 1.0f / vec2(textureSize(uTexInput, 0));
    float result = 0.0;
    vec2 hlim = vec2(float(-u_Uniforms.blurSize) * 0.5 + 0.5);
    for (int i = 0; i < u_Uniforms.blurSize; ++i)
    {
        for (int j = 0; j < u_Uniforms.blurSize; ++j)
        {
            vec2 offset = (hlim + vec2(float(i), float(j))) * texelSize;
            result += texture(uTexInput, texCoords + offset).r;
        }
    }

    AmbientOcclusion = result / float(u_Uniforms.blurSize * u_Uniforms.blurSize);
}