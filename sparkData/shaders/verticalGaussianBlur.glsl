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
};

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
    const vec2 direction = vec2(0.0f, inverseScreenSize.y);

    vec4 color = texture(image, texCoords) * weights[0];
    for(int i = 1; i < 4; ++i)
    {
        color += texture(image, texCoords + direction * float(i)) * weights[i];
        color += texture(image, texCoords - direction * float(i)) * weights[i];
    }

    FragColor = color;
}