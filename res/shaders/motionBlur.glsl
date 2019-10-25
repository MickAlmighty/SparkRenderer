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

#define MAX_SAMPLES 12

void main()
{
    vec3 pos = texture(positionTexture, texCoords).xyz;
    if (pos == vec3(0))
        discard;

    vec2 texelSize = 1.0 / vec2(textureSize(colorTexture, 0));

    vec4 currentPos = viewProjectionMatrix * vec4(pos, 1.0);
    vec4 previousPos = previousViewProjectionMatrix * vec4(pos, 1.0);

    currentPos = currentPos / currentPos.w;
    previousPos = previousPos / previousPos.w;

	float mblurScale = currentFPS / 60.0; // divided by target fps
	vec2 velocity = (currentPos.xy - previousPos.xy) * 0.5;
	velocity *= mblurScale;

	vec3 color = texture(colorTexture, texCoords).rgb;
	float speed = length(velocity / texelSize);
   	
    int numSamples = clamp(int(speed), 1, MAX_SAMPLES);
	for ( uint i = 1; i < numSamples; ++i)
	{
		vec2 offset = velocity * (float(i) / float(numSamples - 1) - 0.5);
		vec3 currentColor = texture(colorTexture, texCoords + offset).rgb;
		color += currentColor;
	}

    FragColor = color / float(numSamples);
}