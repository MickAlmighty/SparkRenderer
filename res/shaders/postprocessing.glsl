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
layout (binding = 1) uniform sampler2D inputTexture2;
uniform vec2 inversedScreenSize;

const float fxaaSpanMax = 8.0f;
const float fxaaReduceMin = 1.0f / 128.0f;
const float fxaaReduceMul = 1.0f / 4.0f;

in vec2 tex_coords;

vec3 FXAA(vec3 color)
{
    vec3 luma = vec3(0.299, 0.587, 0.114);
	float lumaTL = dot(luma, texture(inputTexture, tex_coords + (vec2(-1.0, -1.0) * inversedScreenSize)).rgb);
	float lumaTR = dot(luma, texture(inputTexture, tex_coords + (vec2(1.0, -1.0) * inversedScreenSize)).rgb);
	float lumaBL = dot(luma, texture(inputTexture, tex_coords + (vec2(-1.0, 1.0) * inversedScreenSize)).rgb);
	float lumaBR = dot(luma, texture(inputTexture, tex_coords + (vec2(1.0, 1.0) * inversedScreenSize)).rgb);
	float lumaM = dot(luma, texture(inputTexture, tex_coords).rgb);

	vec2 dir;
	dir.x = -((lumaTL + lumaTR) - (lumaBL + lumaBR));
	dir.y = ((lumaTL + lumaBL) - (lumaTR + lumaBR));

	float dirReduce = max((lumaTL + lumaTR + lumaBL + lumaBR) * (fxaaReduceMul * 0.25), fxaaReduceMin);
	float inverseDirAdjustment = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);

	dir = min(vec2(fxaaSpanMax, fxaaSpanMax),
			  max(vec2(-fxaaSpanMax, -fxaaSpanMax), dir * inverseDirAdjustment)) * inversedScreenSize;

	vec3 result1 = (1.0 / 2.0) * (
		texture(inputTexture, tex_coords + (dir * vec2(1.0 / 3.0 - 0.5))).rgb +
		texture(inputTexture, tex_coords + (dir * vec2(2.0 / 3.0 - 0.5))).rgb);

	vec3 result2 = result1 * (1.0 / 2.0) + (1.0 / 4.0) * (
		texture(inputTexture, tex_coords + (dir * vec2(0.0 / 3.0 - 0.5))).rgb +
		texture(inputTexture, tex_coords + (dir * vec2(3.0 / 3.0 - 0.5))).rgb);

	float lumaMin = min(lumaM, min(min(lumaTL, lumaTR), min(lumaBL, lumaBR)));
	float lumaMax = max(lumaM, max(max(lumaTL, lumaTR), max(lumaBL, lumaBR)));
	float lumaResult2 = dot(luma, result2);

	if (lumaResult2 < lumaMin || lumaResult2 > lumaMax) {
		return result1;
	}
	return result2;

	//return vec4(1, 0, 0, 1); debug
}

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

void main()
{
	//vec3 color = pow(texture(inputTexture, tex_coords).xyz, vec3(2.2f));
	vec3 color = texture(inputTexture, tex_coords).xyz;
	color += texture(inputTexture2, tex_coords).xyz;
	color *= 0.5f;
	color = FXAA(color);
	//vec3 resultColor = uncharted2Tonemap(color);
	vec3 resultColor = reinhardTonemapping(color);
	FragColor = vec4(resultColor, 1);
}