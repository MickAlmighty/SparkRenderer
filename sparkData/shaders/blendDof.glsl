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
layout (location = 0) out vec3 FragColor;

layout (binding = 0) uniform sampler2D cocTexture;
layout (binding = 1) uniform sampler2D sourceColorTexture;
layout (binding = 2) uniform sampler2D bluredColorTexture;

layout (location = 0) in vec2 texCoords;

void main()
{
    vec3 blur = texture(bluredColorTexture, texCoords).xyz;
    vec3 color = texture(sourceColorTexture, texCoords).xyz;
    float cocValue = texture(cocTexture, texCoords).x;

    FragColor = mix(color, blur, cocValue);
}