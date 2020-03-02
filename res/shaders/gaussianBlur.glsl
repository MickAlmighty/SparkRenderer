#type vertex
#version 450
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 textureCoords;

out vec2 texCoords;

void main() {
	texCoords = textureCoords;
	gl_Position = vec4(pos, 1.0);
}

#type fragment
#version 450
layout (location = 0) out vec4 FragColor;

in vec2 texCoords;

uniform sampler2D image;
uniform vec2 inverseScreenSize;
uniform vec2 direction;

const float weights[4] = { 
    0.00598,
    0.060626,
    0.241843,
    0.383103 };
const vec2 offsets[3] = {
    vec2(3.0),
    vec2(2.0),
    vec2(1.0)
};

void main() 
{
	vec4 color = vec4(0.0);

    color += texture(image, texCoords) * weights[3];
    for(int i = 0; i < 3; ++i)
    {
        //horizontal blur
        color += texture(image, texCoords + offsets[i] * inverseScreenSize * direction) * weights[i];
        color += texture(image, texCoords - offsets[i] * inverseScreenSize * direction) * weights[i];
    }

	//color.a = 1.0f;
    FragColor = color;
}
