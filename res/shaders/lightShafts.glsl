#type vertex
#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 textureCoords;

out vec2 texCoords;

void main()
{
    texCoords = textureCoords;
    gl_Position = vec4(position, 1);
}

#type fragment
#version 450
layout (location = 0) out vec4 FragColor;

in vec2 texCoords;

layout (binding = 0) uniform sampler2D depthTexture;

uniform vec2 lightScreenPos;
uniform vec3 lightColor;

uniform int samples = 100;
uniform float exposure = 0.0034f;
uniform float decay = 0.995f;
uniform float density = 0.75f;
uniform float weight = 6.65;

void main()
{
    vec4 outColor = vec4(0);

    vec2 textureCoords = texCoords;
	vec2 texCoordStep = (textureCoords - lightScreenPos);
    texCoordStep *= (1.0 / float(samples)) * density;

	float distanceToScreenCenter = clamp(1.0f - length(vec2(0.5f) - lightScreenPos) * 2.5f, 0.0f, 1.0f);

    float illuminationDecay = 1.0f;

    for(int i = 0; i < samples; ++i)
	{
		// Step sample location along ray.
		textureCoords -= texCoordStep;
 		
		vec4 colorSample =  vec4(lightColor, 1.0f);

		float depth = texture(depthTexture, clamp(textureCoords, 0.0f, 1.0f)).x;
		if (depth != 0.0f)
		{
			colorSample = vec4(0);
		}
		
		// Apply sample attenuation scale/decay factors.
		colorSample  *= illuminationDecay * weight;
 
		// Accumulate combined color.  
		outColor += colorSample;
 
		// Update exponential decay factor.
		illuminationDecay *= decay;
	}

    FragColor = outColor * exposure * distanceToScreenCenter;
}