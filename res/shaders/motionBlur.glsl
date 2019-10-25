#type vertex
#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texture_coords;

out vec2 texCoords;

void main()
{
    texCoords = texture_coords;
    gl_Position = vec4(position, 1.0);
}

#type fragment
#version 450

layout (location = 0) out vec3 FragColor;

in vec2 texCoords;

layout (binding = 0) uniform sampler2D colorTexture;
layout (binding = 1) uniform sampler2D positionTexture;

uniform mat4 viewProjectionMatrix;
uniform mat4 previousViewProjectionMatrix;
uniform float currentFPS;

void main()
{
    vec3 pos = texture(positionTexture, texCoords).xyz;
    if (pos == vec3(0))
        discard;

    vec2 texelSize = 1.0 / vec2(textureSize(colorTexture, 0));

    vec4 currentPos = viewProjectionMatrix * vec4(pos, 1.0);
    vec4 previousPos = previousViewProjectionMatrix * vec4(pos, 1.0);

    currentPos = currentPos / currentPos.w;
	//currentPos = currentPos * 0.5 + 0.5;

    previousPos = previousPos / previousPos.w;
	//previousPos = previousPos * 0.5 + 0.5;

	float mblurScale = currentFPS / 60.0; // divided by target fps
	vec2 velocity = (currentPos.xy - previousPos.xy) / 2.0;
	velocity *= mblurScale;

	vec3 color = texture(colorTexture, texCoords).rgb;
	int numSamples = 16;
	
	for ( uint i = 1; i < numSamples; ++i)
	{
		vec2 offset = velocity * (float(i) / float(numSamples - 1) - 0.5);
		vec3 currentColor = texture(colorTexture, texCoords + offset).rgb;
		color += currentColor;
	}

    FragColor = color / float(numSamples);
}