#type vertex
#version 450
layout (location = 0) in vec3 Position;
layout (location = 1) in vec2 TextureCoords;

out vec2 texCoords;

void main()
{
    texCoords = TextureCoords;
    gl_Position = vec4(Position, 1.0f);
}

#type fragment
#version 450
layout ( binding = 0, offset = 0) uniform atomic_uint BokehCounter;
layout ( binding = 1) writeonly uniform image1D BokehPositionTex;
layout ( binding = 2) writeonly uniform image1D BokehColorTex;

layout (binding = 3) uniform sampler2D colorTexture;
layout (binding = 4) uniform sampler2D cocTexture;

uniform float CoCThreshold = 0.1f; 
uniform float LumThreshold = 1.0f;

in vec2 texCoords;

void main() 
{
    float coc = texture(cocTexture, texCoords).x;
    if (coc == 0)
        discard;

	vec3 colorNeighs = vec3(0.0f);
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(-2, -2), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(-2, -1), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(-2, 0), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(-2, 1), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(-2, 2), 0.0f).xyz;

	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(-1, -2), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(-1, -1), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(-1, 0), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(-1, 1), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(-1, 2), 0.0f).xyz;

	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(0, -2), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(0, -1), 0.0f).xyz;
	//without (0,0) -> center
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(0, 1), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(0, 2), 0.0f).xyz;

	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(1, -2), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(1, -1), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(1, 0), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(1, 1), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(1, 2), 0.0f).xyz;

	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(2, -2), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(2, -1), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(2, 0), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(2, 1), 0.0f).xyz;
	colorNeighs += textureOffset(colorTexture, texCoords, ivec2(2, 2), 0.0f).xyz;

	colorNeighs /= 24.0f;
	vec3 colorCenter = texture(colorTexture, texCoords).xyz;

	// Append pixel whose constrast is greater than the user's threshold
	float lumNeighs = dot(colorNeighs , vec3 (0.299f, 0.587f, 0.114f));
	float lumCenter = dot(colorCenter , vec3 (0.299f, 0.587f, 0.114f));

	if((lumCenter - lumNeighs) > LumThreshold && coc > CoCThreshold)
	{
		int current = int(atomicCounterIncrement(BokehCounter));
        if (current > 1023)
            return;
		imageStore(BokehPositionTex, current, vec4(texCoords.x, texCoords.y, coc , coc));
		imageStore(BokehColorTex, current, vec4(colorCenter, 1));
	}
}