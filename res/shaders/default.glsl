#type vertex
#version 450
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texture_coords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 tex_coords;

out TO_TEXTURES {
	vec3 position;
	vec3 normal;
} to_textures;

void main()
{
	vec4 worldPosition = model * vec4(position, 1);

	mat3 normalMatrix = transpose(inverse(mat3(model)));
	to_textures.normal = normalMatrix * normal;
	to_textures.position = worldPosition.xyz;
   
	tex_coords = texture_coords;

	gl_Position = projection * view * worldPosition;
}

#type fragment
#version 450
layout (location = 0) out vec3 Position;
layout (location = 1) out vec4 FragColor;
layout (location = 2) out vec3 Normal;
layout (location = 3) out float Roughness;
layout (location = 4) out float Metalness;

layout (binding = 1) uniform sampler2D diffuseTexture;
layout (binding = 2) uniform sampler2D normalTexture;
layout (binding = 3) uniform sampler2D roughnessTexture;
layout (binding = 4) uniform sampler2D metalnessTexture;

in vec2 tex_coords;

in TO_TEXTURES {
	vec3 position;
	vec3 normal;
} to_textures;

void main()
{
	Position = to_textures.position;
    FragColor = pow(texture(diffuseTexture, tex_coords), vec4(2.2));
    Normal = normalize(to_textures.normal);
	//Normal = normalize(texture(normalTexture, tex_coords).xyz);// *0.5 + 0.5); // transforms from [-1,1] to [0,1]  
	Roughness = texture(roughnessTexture, tex_coords).x;
	Metalness = texture(metalnessTexture, tex_coords).x;
}