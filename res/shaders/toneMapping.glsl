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
layout (location = 0) out vec3 FragColor;

layout (binding = 0) uniform sampler2D inputTexture;
layout (binding = 1) uniform sampler2D averageLuminance;
uniform vec2 invertedScreenSize;

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
	return color / (color + vec3(1));
}

vec3 ACESFilm(vec3 x)
{
	float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
	return  clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

vec3 approximationLinearToSRGB(vec3 linearColor)
{
    return pow(linearColor, vec3(1.0f / 2.2f));
}

vec3 accurateLinearToSRGB(vec3 linearColor)
{
	// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
	// page 88
    vec3 sRGBLo = linearColor * 12.92f;
    vec3 sRGBHi = (pow(abs(linearColor), vec3(1.0f / 2.4)) * 1.055f) - 0.055f;
    vec3 sRGB;
    sRGB.x = (linearColor.x <= 0.0031308f) ? sRGBLo.x : sRGBHi.x;
    sRGB.y = (linearColor.y <= 0.0031308f) ? sRGBLo.y : sRGBHi.y;
    sRGB.z = (linearColor.z <= 0.0031308f) ? sRGBLo.z : sRGBHi.z;
    return sRGB;
}

void main()
{
	vec3 color = texture(inputTexture, tex_coords).xyz; //for unchartedTonemapping
	float avgLuminance = texture(averageLuminance, tex_coords).x;

	color /= avgLuminance;
	vec3 resultColor = approximationLinearToSRGB(ACESFilm(color));
	//resultColor = approximationLinearToSRGB(resultColor);
	FragColor = resultColor;
}