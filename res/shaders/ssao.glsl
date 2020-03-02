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
layout (location = 0) out float AmbientOcclusion;

layout (binding = 0) uniform sampler2D depthTexture;
layout (binding = 1) uniform sampler2D normalTexture;
layout (binding = 2) uniform sampler2D texNoise;

layout (std140) uniform Camera
{
	vec4 pos;
	mat4 view;
	mat4 projection;
	mat4 invertedView;
	mat4 invertedProjection;
} camera;

layout (std140) uniform Samples
{
	vec3 samples[64];
};

uniform int kernelSize = 32;
uniform float radius = 0.3f;
uniform float bias = 0.01f;
uniform float power = 1.0f;
uniform vec2 screenSize = vec2(1280.0f, 720.0f);

in vec2 texCoords;

vec4 viewPosFromDepth(float depth, mat4 invProj, vec2 uv) 
{
    vec4 clipSpacePosition = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewSpacePosition = invProj * clipSpacePosition;

    // Perspective division
    viewSpacePosition /= viewSpacePosition.w;

    return viewSpacePosition;
}

vec3 decode(vec2 enc)
{
	vec2 fenc = enc * 4.0f - 2.0f;
    float f = dot(fenc, fenc);
    float g = sqrt(1.0f - f / 4.0f);
    vec3 n;
    n.xy = fenc*g;
    n.z = 1.0f - f / 2.0f;
    return n;
}

vec3 getViewSpacePosition(vec2 uv)
{
	float depth = texture(depthTexture, uv).x;
	return viewPosFromDepth(depth, camera.invertedProjection, uv).xyz;
}

vec3 getNormal(vec2 uv)
{
	return decode(texture(normalTexture, texCoords).xy);
}

void main() 
{
    float depthValue = texture(depthTexture, texCoords).x;
	if (depthValue == 0.0f)
	{
		discard;
	}

	const vec2 noiseScale = vec2(screenSize.x / 4.0f, screenSize.y / 4.0f);

	vec3 P = getViewSpacePosition(texCoords);
	vec3 N = getNormal(texCoords);
	vec3 randomVec = normalize(texture(texNoise, texCoords * noiseScale).xyz);

	float occlusion = 0.0f;

	for(int i = 0; i < kernelSize; ++i)
    {
        // get sample position
        vec3 ray = radius * reflect(samples[i], randomVec);
        vec3 sampleP = P + sign(dot(ray, N)) * ray;
        
        // project sample position (to sample texture) (to get position on screen/texture)
        vec4 offset = vec4(sampleP, 1.0f);
        offset = camera.projection * offset; // from view to clip-space
        offset.xyz /= offset.w; // perspective divide
        offset.xy = offset.xy * 0.5f + 0.5f; // transform to range 0.0 - 1.0
        
        // get sample depth
        vec3 sampleViewPos = getViewSpacePosition(offset.xy);
        // range check & accumulate
        float rangeCheck = smoothstep(0.0f, 1.0f, radius / abs(P.z - sampleViewPos.z));
        occlusion += (sampleViewPos.z >= sampleP.z + bias ? 1.0f : 0.0f) * rangeCheck;           
    }
    occlusion = 1.0f - (occlusion / kernelSize);
    AmbientOcclusion.x = pow(occlusion, power);
}