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

layout (binding = 0) uniform sampler2D image;
uniform bool horizontal = true;
uniform vec2 inverseScreenSize;

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
	// vec2 texSize = textureSize(image, 0);
	// vec2 texStep = 1.0f / texSize;
	vec2 texStep = inverseScreenSize;
	
    vec2 direction = vec2(0.0f);
	if (horizontal == true)
	{
		direction.x = 1.0f * texStep.x;
	}
	else
	{
		direction.y = 1.0f  * texStep.y;
	}

    //vec4 color = vec4(0.0);
    vec4 color = texture(image, texCoords) * weights[0];
    for(int i = 1; i < 4; ++i)
    {
        color += texture(image, texCoords + direction * float(i)) * weights[i];
        color += texture(image, texCoords - direction * float(i)) * weights[i];
    }

    FragColor = color;
}