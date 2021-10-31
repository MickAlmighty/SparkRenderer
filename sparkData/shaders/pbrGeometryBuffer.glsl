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

out VS_OUT {
    vec2 tex_coords;
    mat3 viewTBN;
    vec3 tangentFragPos;
    vec3 tangentCamPos;
    vec3 normalView;
} vs_out;

void main()
{
    vec4 worldPosition = model * vec4(position, 1);

    mat3 normalMatrix = mat3(transpose(inverse(model)));
    vec3 T = normalize(normalMatrix * tangent);
    vec3 N = normalize(normalMatrix * normal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);
    mat3 viewTBN = mat3(camera.view) * TBN;
    mat3 inverseTBN = transpose(TBN);

    vs_out.tangentFragPos = inverseTBN * worldPosition.xyz;
    vs_out.tangentCamPos  = inverseTBN * camera.pos.xyz;
    vs_out.tex_coords = texture_coords;
    vs_out.viewTBN = viewTBN;
    mat4 viewModel = camera.view * model;
    vs_out.normalView = vec3(viewModel * vec4(normal, 0));

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
layout (binding = 5) uniform sampler2D heightTexture;

in VS_OUT {
    vec2 tex_coords;
    mat3 viewTBN;
    vec3 tangentFragPos;
    vec3 tangentCamPos;
    vec3 normalView;
} vs_out;

const float heightScale = 0.05f;

vec2 parallaxMapping(vec2 texCoords, vec3 viewDir)
{
    // number of depth layers
    const float minLayers = 8.0;
    const float maxLayers = 32.0;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), viewDir)));
    // calculate the size of each layer
    float layerDepth = 1.0 / numLayers;
    // depth of current layer
    float currentLayerDepth = 0.0;
    // the amount to shift the texture coordinates per layer (from vector P)
    vec2 P = viewDir.xy * heightScale;
    vec2 deltaTexCoords = P / numLayers;

    vec2  currentTexCoords = texCoords;
    float currentDepthMapValue = 1.0f - texture(heightTexture, currentTexCoords).r;

    while (currentLayerDepth < currentDepthMapValue)
    {
        // shift texture coordinates along direction of P
        currentTexCoords -= deltaTexCoords;
        // get depthmap value at current texture coordinates
        currentDepthMapValue = 1.0f - texture(heightTexture, currentTexCoords).r;
        // get depth of next layer
        currentLayerDepth += layerDepth;
    }

    vec2 prevTexCoords = currentTexCoords + deltaTexCoords;

    // get depth after and before collision for linear interpolation
    float afterDepth = currentDepthMapValue - currentLayerDepth;
    float beforeDepth = (1.0f - texture(heightTexture, prevTexCoords).r) - currentLayerDepth + layerDepth;

    // interpolation of texture coordinates
    float weight = afterDepth / (afterDepth - beforeDepth);
    vec2 finalTexCoords = prevTexCoords * weight + currentTexCoords * (1.0 - weight);

    return finalTexCoords;
}

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

vec3 getViewNormal(vec2 tc)
{
    vec3 normalFromTexture = texture(normalTexture, tc).xyz;
    if (normalFromTexture.xy != vec2(0))
    {
        if (normalFromTexture.z == 0)
        {
            normalFromTexture = normalize(normalFromTexture * 2.0 - 1.0);
            vec2 nXY = normalFromTexture.xy;
            normalFromTexture.z = sqrt(1.0f - (nXY.x * nXY.x) - (nXY.y * nXY.y));
            return normalize(vs_out.viewTBN * normalFromTexture);
        }

        normalFromTexture = normalize(normalFromTexture * 2.0 - 1.0);
        vec3 viewNormal = normalize(vs_out.viewTBN * normalFromTexture);
        return viewNormal;
    }
    else
    {
        vec3 viewNormal = normalize(vs_out.normalView);
        return viewNormal;
    }
}

void main()
{
    vec2 tex_coords = vs_out.tex_coords;
    if (texture(heightTexture, vs_out.tex_coords).r != 0.0) 
    {
        vec3 tangentViewDir = normalize(vs_out.tangentCamPos - vs_out.tangentFragPos);
        tex_coords = parallaxMapping(vs_out.tex_coords, tangentViewDir);
    }

    //if (tex_coords.x > 1.0 || tex_coords.x < 0.0 || tex_coords.y > 1.0 || tex_coords.y < 0.0)
    //    discard;

    FragColor.rgb = accurateSRGBToLinear(texture(diffuseTexture, tex_coords).rgb);

    vec3 viewNormal = getViewNormal(tex_coords);
    vec2 encodedNormal = encodeViewSpaceNormal(viewNormal);
    Normal.rg = encodedNormal;

    RoughnessMetalness.r = texture(roughnessTexture, tex_coords).x;
    RoughnessMetalness.g = texture(metalnessTexture, tex_coords).x;
}