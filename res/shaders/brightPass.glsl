#type vertex
#version 450
layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 textureCoords;

out vec2 texCoords;

void main()
{
    texCoords = textureCoords;
    gl_Position = vec4(pos, 1.0);
}

#type fragment
#version 450
layout (location = 0) out vec3 BrightColor;

layout (binding = 0) uniform sampler2D lightTexture;

in vec2 texCoords;

void main()
{
    const float brightLimit = 1.0f;
    vec3 lightColor = texture(lightTexture, texCoords).xyz;

    // vec3 brightColor = max(lightColor - vec3(brightLimit), vec3(0.0));
    // float bright = dot(brightColor, vec3(1.0));
    // bright = smoothstep(0.0, 0.5, bright);
    // BrightColor = mix(vec3(0.0), lightColor, bright);
    BrightColor = lightColor * 0.1;
}