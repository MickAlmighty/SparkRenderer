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
layout (location = 0) out vec3 FragColor;

layout (binding = 0) uniform sampler2D inputTexture;
uniform vec2 inversedScreenSize;

in vec2 texCoords;

const float fxaaSpanMax = 8.0f;
const float fxaaReduceMin = 1.0f / 128.0f;
const float fxaaReduceMul = 1.0f / 8.0f;

vec3 FXAA();

void main() 
{
    FragColor = FXAA();
}

vec3 FXAA()
{
    vec3 luma = vec3(0.299, 0.587, 0.114);
	float lumaTL = dot(luma, texture(inputTexture, texCoords + (vec2(-1.0, -1.0) * inversedScreenSize)).rgb);
	float lumaTR = dot(luma, texture(inputTexture, texCoords + (vec2(1.0, -1.0) * inversedScreenSize)).rgb);
	float lumaBL = dot(luma, texture(inputTexture, texCoords + (vec2(-1.0, 1.0) * inversedScreenSize)).rgb);
	float lumaBR = dot(luma, texture(inputTexture, texCoords + (vec2(1.0, 1.0) * inversedScreenSize)).rgb);
	float lumaM = dot(luma, texture(inputTexture, texCoords).rgb);

	vec2 dir;
	dir.x = -((lumaTL + lumaTR) - (lumaBL + lumaBR));
	dir.y = ((lumaTL + lumaBL) - (lumaTR + lumaBR));

	float dirReduce = max((lumaTL + lumaTR + lumaBL + lumaBR) * (fxaaReduceMul * 0.25), fxaaReduceMin);
	float inverseDirAdjustment = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);

	dir = min(vec2(fxaaSpanMax, fxaaSpanMax),
			  max(vec2(-fxaaSpanMax, -fxaaSpanMax), dir * inverseDirAdjustment)) * inversedScreenSize;

	vec3 result1 = (1.0 / 2.0) * (
		texture(inputTexture, texCoords + (dir * vec2(1.0 / 3.0 - 0.5))).rgb +
		texture(inputTexture, texCoords + (dir * vec2(2.0 / 3.0 - 0.5))).rgb);

	vec3 result2 = result1 * (1.0 / 2.0) + (1.0 / 4.0) * (
		texture(inputTexture, texCoords + (dir * vec2(0.0 / 3.0 - 0.5))).rgb +
		texture(inputTexture, texCoords + (dir * vec2(3.0 / 3.0 - 0.5))).rgb);

	float lumaMin = min(lumaM, min(min(lumaTL, lumaTR), min(lumaBL, lumaBR)));
	float lumaMax = max(lumaM, max(max(lumaTL, lumaTR), max(lumaBL, lumaBR)));
	float lumaResult2 = dot(luma, result2);

	if (lumaResult2 < lumaMin || lumaResult2 > lumaMax) 
	{
		return result1;
	}
	return result2;
    //return vec3(1, 0, 0); // debug
}