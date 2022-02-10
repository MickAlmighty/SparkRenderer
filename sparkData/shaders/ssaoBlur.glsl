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
layout (location = 0) out float AmbientOcclusion;

layout (location = 0) in vec2 texCoords;

layout (location = 0) uniform int uBlurSize = 4; // use size of noise texture
layout (binding = 0) uniform sampler2D uTexInput;

void main() 
{
    vec2 texelSize = 1.0f / vec2(textureSize(uTexInput, 0));
    float result = 0.0;
    vec2 hlim = vec2(float(-uBlurSize) * 0.5 + 0.5);
    for (int i = 0; i < uBlurSize; ++i)
    {
        for (int j = 0; j < uBlurSize; ++j)
        {
            vec2 offset = (hlim + vec2(float(i), float(j))) * texelSize;
            result += texture(uTexInput, texCoords + offset).r;
        }
    }

    AmbientOcclusion = result / float(uBlurSize * uBlurSize);
}