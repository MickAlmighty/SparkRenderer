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
layout (location = 0) out vec3 FragColor;
layout (location = 1) out vec2 Normal;
layout (location = 2) out vec2 RoughnessMetalness;

layout (binding = 1) uniform sampler2D diffuseTexture;
layout (binding = 2) uniform sampler2D normalTexture;
layout (binding = 3) uniform sampler2D roughnessTexture;
layout (binding = 4) uniform sampler2D metalnessTexture;

in vec2 tex_coords;
in mat3 viewTBN_matrix;

vec2 encodeViewSpaceNormal(vec3 n)
{
    //Lambert Azimuthal Equal-Area projection
    //http://aras-p.info/texts/CompactNormalStorage.html
    float p = sqrt(n.z*8+8);
    return vec2(n.xy/p + 0.5);
}

vec3 approximationSRgbToLinear (vec3 sRGBColor )
{
    return pow ( sRGBColor, vec3(2.2));
}

vec3 accurateSRGBToLinear(vec3 sRGBColor)
{
    // https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    // page 88
    vec3 linearRGBLo = sRGBColor / 12.92f;
    vec3 linearRGBHi = pow((sRGBColor + 0.055f) / 1.055f, vec3(2.4f));
    vec3 linearRGB;
    linearRGB.x = (sRGBColor.x <= 0.04045f) ? linearRGBLo.x : linearRGBHi.x;
    linearRGB.y = (sRGBColor.y <= 0.04045f) ? linearRGBLo.y : linearRGBHi.y;
    linearRGB.z = (sRGBColor.z <= 0.04045f) ? linearRGBLo.z : linearRGBHi.z;
    return linearRGB;
}

void main()
{
    FragColor.rgb = accurateSRGBToLinear(texture(diffuseTexture, tex_coords).rgb);

    vec3 normalFromTexture = texture(normalTexture, tex_coords).xyz;
    normalFromTexture = normalize(normalFromTexture * 2.0 - 1.0);
    vec3 viewNormal = normalize(viewTBN_matrix * normalFromTexture);
    vec2 encodedNormal = encodeViewSpaceNormal(viewNormal);
    Normal.rg = encodedNormal;

    RoughnessMetalness.r = texture(roughnessTexture, tex_coords).x;
    RoughnessMetalness.g = texture(metalnessTexture, tex_coords).x;
}