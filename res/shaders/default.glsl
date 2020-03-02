#type vertex
#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texture_coords;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;

uniform mat4 model;

layout (std140) uniform Camera
{
	vec4 pos;
	mat4 view;
	mat4 projection;
	mat4 invertedView;
	mat4 invertedProjection;
} camera;

out vec2 tex_coords;
out mat3 viewTBN_matrix;

void main()
{
	vec4 worldPosition = model * vec4(position, 1);

	mat3 normalMatrix = transpose(inverse(mat3(model)));
	vec3 T = normalize(normalMatrix * tangent);
	vec3 N = normalize(normalMatrix * normal);
	//T = normalize(T - dot(T, N) * N);
	//vec3 B = cross(N, T);
	vec3 B = normalize(normalMatrix * bitangent);
	mat3 viewTBN = mat3(camera.view) * mat3(T, B, N);
	viewTBN_matrix = viewTBN;
   
	tex_coords = texture_coords;

	gl_Position = camera.projection * camera.view * worldPosition;
}

#type fragment
#version 450
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec3 Normal;

layout (binding = 1) uniform sampler2D diffuseTexture;
layout (binding = 2) uniform sampler2D normalTexture;
layout (binding = 3) uniform sampler2D roughnessTexture;
layout (binding = 4) uniform sampler2D metalnessTexture;

in vec2 tex_coords;
in mat3 viewTBN_matrix;

vec2 encode(vec3 n)
{
	float p = sqrt(n.z*8+8);
    return vec2(n.xy/p + 0.5);
}

// vec3 decode(vec2 enc)
// {
// 	vec2 fenc = enc*4-2;
//     float f = dot(fenc,fenc);
//     float g = sqrt(1-f/4);
//     vec3 n;
//     n.xy = fenc*g;
//     n.z = 1-f/2;
//     return n;
// }

void main()
{
    FragColor.xyz = texture(diffuseTexture, tex_coords).xyz;

	vec3 normalFromTexture = texture(normalTexture, tex_coords).xyz;
	normalFromTexture = normalize(normalFromTexture * 2.0 - 1.0);

	vec3 viewNormal = normalize(viewTBN_matrix * normalFromTexture);

	vec2 encodedNormal = encode(viewNormal);
	Normal.xy = encodedNormal;
	Normal.z = texture(metalnessTexture, tex_coords).x;
	//Normal.xyz = normalize(TBN_matrix * normalFromTexture);
	FragColor.w = texture(roughnessTexture, tex_coords).x;
}