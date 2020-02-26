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
	//FragColor = vec4(distanceToScreenCenter);
	//return;
    float illuminationDecay = 1.0f;

    for(int i = 0; i < samples; ++i)
	{
		// Step sample location along ray.
		textureCoords -= texCoordStep;
		vec4 colorSample = vec4(0.0f);
 		// float lightRadius = length(textureCoords - lightScreenPos);
		// if (lightRadius < 0.1f)
		// {
		// 	colorSample = vec4(0.3f) * vec4(lightColor, 1.0f);
		// }
		// else
		// {
			colorSample = vec4(1) * vec4(lightColor, 1.0f);
		//}

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

	//if (texture(depthTexture, texCoords).x != 0.0f)
    	FragColor = outColor * exposure * distanceToScreenCenter;
	//else 
	//	FragColor = outColor * exposure * 0.1f;
    //FragColor = vec4(textureCoords.x, textureCoords.y, illuminationDecay, 0.0f);
}