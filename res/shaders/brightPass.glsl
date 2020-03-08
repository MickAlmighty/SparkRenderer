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

uniform float brightThreshold = 3.0f;
uniform float brightThresholdRange = 1000.0f;

const vec3 luminance = vec3(0.2126f, 0.7152f, 0.0722f); 
const vec3 luminance2 = vec3(0.299f, 0.587f, 0.114f); 


float almostIdentity(float x)
{
    return x * x * (2.0 - x);
}

float linearDistance(float x, float near, float far)
{
    return clamp((x - near) / (far - near), 0.0f, 1.0f); //clamp distance to linear values between [0,1]
}

vec3 ACESFilm(vec3 x)
{
	float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
	vec3 color = clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
	return color;
}

const float power = 4.0f;
const float scale = 1.83f;
const float bias = 0.27f;

vec3 brightPass(vec3 color)
{
    return (-color + pow(color, vec3(power)) * scale) / bias;
}

void main()
{
    vec3 lightColor = texture(lightTexture, texCoords).xyz;
    BrightColor = brightPass(lightColor);
	//float brightness = dot(lightColor, luminance);
    vec3 col = ACESFilm(lightColor);
	if (length(col) > 0.9f)
	{

        BrightColor = col;//ACESFilm(lightColor);
        //BrightColor = vec3(distance);
	}
	else
		BrightColor = vec3(0.0f);
}