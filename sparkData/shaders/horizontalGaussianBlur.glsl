#type vertex
#version 450
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 textureCoords;

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

layout (binding = 0) uniform sampler2D image;

layout (push_constant) uniform PushConstants
{
    vec2 inverseScreenSize;
} u_Uniforms;

const float weights[4] = {
    0.383103,
    0.241843,
    0.060626,
    0.00598 + 0.000229
    // last weight = 0.000229 merged into previous
};

const float weights5x5[3] = {
    0.38774f,
    0.24477f,
    0.06136f
};

void main() 
{
    const vec2 direction = vec2(u_Uniforms.inverseScreenSize.x, 0.0f);

    const vec2 texSize = vec2(textureSize(image, 0).xy);
    const vec2 screenRatioScaling = vec2(texSize.y / texSize.x, 1.0f);

    vec4 color = texture(image, texCoords) * weights[0];
    for(int i = 1; i < 4; ++i)
    {
        const vec2 offset = direction * screenRatioScaling * float(i);
        color += texture(image, texCoords + offset) * weights[i];
        color += texture(image, texCoords - offset) * weights[i];
    }

    FragColor = color;
}