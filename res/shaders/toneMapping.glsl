#type vertex
#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coords;

out vec2 tex_coords;

void main()
{
    tex_coords = texture_coords;
    gl_Position = vec4(position, 1);
}

#type fragment
#version 450
layout (location = 0) out vec4 FragColor;

layout (binding = 0) uniform sampler2D inputTexture;
uniform vec2 inversedScreenSize;

in vec2 tex_coords;

float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

vec3 tonemapCalculations(vec3 x)
{
	return ((x*(A*x + C * B) + D * E) / (x*(A*x + B) + D * F)) - E / F;
}

vec3 uncharted2Tonemap(vec3 inColor)
{
	//inColor *= 16.0f;

	float exposureBias = 2.0f;
	vec3 curr = tonemapCalculations(exposureBias * inColor);

	vec3 whiteScale = vec3(1.0f) / tonemapCalculations(vec3(W));
	vec3 color = curr * whiteScale;

	return pow(color, vec3(1 / 2.2f));
}

vec3 reinhardTonemapping(vec3 color)
{
	return pow(color / (color + vec3(1)), vec3(1 / 2.2f));
}

vec3 ACESFilm(vec3 x)
{
	float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
	vec3 color = clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
	return pow(color, vec3(1 / 2.2f));
}

void main()
{
	vec3 color = texture(inputTexture, tex_coords).xyz; //for unchartedTonemapping
	//vec3 resultColor = uncharted2Tonemap(color);
	//vec3 resultColor = reinhardTonemapping(color);
	vec3 resultColor = ACESFilm(color);
	FragColor = vec4(resultColor, 1);
}