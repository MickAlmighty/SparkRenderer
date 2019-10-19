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

uniform float u_lumaThreshold = 0.4;
uniform float u_mulReduce = 1.0/8.0;
uniform float u_minReduce = 1.0/128.0;
uniform float u_maxSpan = 8.0;

in vec2 tex_coords;

vec4 FXAA(vec4 color)
{
    //color = pow(color, vec4(2.2));
	vec3 rgbM = texture(inputTexture, tex_coords).rgb;

    vec3 rgbNW = textureOffset(inputTexture, tex_coords, ivec2(-1, 1)).rgb;
    vec3 rgbNE = textureOffset(inputTexture, tex_coords, ivec2(1, 1)).rgb;
    vec3 rgbSW = textureOffset(inputTexture, tex_coords, ivec2(-1, -1)).rgb;
    vec3 rgbSE = textureOffset(inputTexture, tex_coords, ivec2(1, -1)).rgb;

	const vec3 toLuma = vec3(0.299, 0.587, 0.114);

	// Convert from RGB to luma.
	float lumaNW = dot(rgbNW, toLuma);
	float lumaNE = dot(rgbNE, toLuma);
	float lumaSW = dot(rgbSW, toLuma);
	float lumaSE = dot(rgbSE, toLuma);
	float lumaM = dot(rgbM, toLuma);

	// Gather minimum and maximum luma.
	float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
	float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

	if (lumaMax - lumaMin <= lumaMax * u_lumaThreshold)
	{
		// ... do no AA and return.
		return vec4(rgbM, 1.0);
	}

	// Sampling is done along the gradient.
	vec2 samplingDirection;	
	samplingDirection.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    samplingDirection.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

	// Sampling step distance depends on the luma: The brighter the sampled texels, the smaller the final sampling step direction.
    // This results, that brighter areas are less blurred/more sharper than dark areas.  
    float samplingDirectionReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * u_mulReduce, u_minReduce);

	// Factor for norming the sampling direction plus adding the brightness influence. 
	float minSamplingDirectionFactor = 1.0 / (min(abs(samplingDirection.x), abs(samplingDirection.y)) + samplingDirectionReduce);

	// Calculate final sampling direction vector by reducing, clamping to a range and finally adapting to the texture size. 
    samplingDirection = clamp(samplingDirection * minSamplingDirectionFactor, vec2(-u_maxSpan), vec2(u_maxSpan)) * inversedScreenSize;

	// Inner samples on the tab.
	vec3 rgbSampleNeg = texture(inputTexture, tex_coords + samplingDirection * (1.0/3.0 - 0.5)).rgb;
	vec3 rgbSamplePos = texture(inputTexture, tex_coords + samplingDirection * (2.0/3.0 - 0.5)).rgb;

	vec3 rgbTwoTab = (rgbSamplePos + rgbSampleNeg) * 0.5;  

	// Outer samples on the tab.
	vec3 rgbSampleNegOuter = texture(inputTexture, tex_coords + samplingDirection * (0.0/3.0 - 0.5)).rgb;
	vec3 rgbSamplePosOuter = texture(inputTexture, tex_coords + samplingDirection * (3.0/3.0 - 0.5)).rgb;

	vec3 rgbFourTab = (rgbSamplePosOuter + rgbSampleNegOuter) * 0.25 + rgbTwoTab * 0.5;   
	
	// Calculate luma for checking against the minimum and maximum value.
	float lumaFourTab = dot(rgbFourTab, toLuma);
	
	//Are outer samples of the tab beyond the edge ... 
	if (lumaFourTab < lumaMin || lumaFourTab > lumaMax)
	{
		// ... yes, so use only two samples.
		return vec4(rgbTwoTab, 1.0); 
	}
	else
	{
		// ... no, so use four samples. 
		return vec4(rgbFourTab, 1.0);
	}


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
	//vec3 fxaaColor = FXAA(vec4(color, 1)).xyz;
	vec3 resultColor = uncharted2Tonemap(color);
	//vec3 resultColor = reinhardTonemapping(color);
    FragColor = vec4(resultColor, 1);
}