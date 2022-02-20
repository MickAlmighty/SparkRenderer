#type vertex
#version 450
layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 textureCoords;

layout (location = 0) out vec2 texCoords;

void main()
{
    texCoords = textureCoords;
    gl_Position = vec4(pos, 1.0);
}

#type fragment
#version 450
layout (location = 0) in vec2 texCoords;
layout (location = 0) out vec4 FragColor;

layout (binding = 0) uniform sampler2D inputTexture;

layout (push_constant) uniform PushConstants
{
    float intensity;
    vec2 outputTextureSizeInversion; //equals to 1.0f / texture size 
    float radius;
} u_Uniforms;

vec2 texelOffset(vec2 offset)
{
    return offset * u_Uniforms.outputTextureSizeInversion * u_Uniforms.radius;
}

#define weight 1.0 / 16.0

void main()
{
    vec3 sample1 = texture(inputTexture, texCoords + texelOffset(vec2(-1.0f, 1.0f))).rgb;
    vec3 sample2 = texture(inputTexture, texCoords + texelOffset(vec2(0.0f, 1.0f))).rgb * 2.0f;
    vec3 sample3 = texture(inputTexture, texCoords + texelOffset(vec2(1.0f, 1.0f))).rgb;
    vec3 sample4 = texture(inputTexture, texCoords + texelOffset(vec2(-1.0f, 0.0f))).rgb * 2.0f;
    vec3 sample5 = texture(inputTexture, texCoords + texelOffset(vec2(0.0f, 0.0f))).rgb * 4.0f;
    vec3 sample6 = texture(inputTexture, texCoords + texelOffset(vec2(1.0f, 0.0f))).rgb * 2.0f;
    vec3 sample7 = texture(inputTexture, texCoords + texelOffset(vec2(-1.0f, -1.0f))).rgb;
    vec3 sample8 = texture(inputTexture, texCoords + texelOffset(vec2(0.0f, -1.0f))).rgb * 2.0f;
    vec3 sample9 = texture(inputTexture, texCoords + texelOffset(vec2(1.0f, -1.0f))).rgb;
    vec3 sum = (sample1 + sample2 + sample3 + sample4 + sample5 + sample6 + sample7 + sample8 + sample9) * weight;
    sum *= max(u_Uniforms.intensity, 0.0f);
    FragColor = vec4(sum, 0);
}