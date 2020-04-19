#type vertex
#version 450
layout (location = 0) in vec3 aPos;

out vec3 localPos;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    localPos = aPos;  
    gl_Position =  projection * view * vec4(localPos, 1.0);
}

#type fragment
#version 450 core
layout (location = 0) out vec4 FragColor;

layout (location = 0) in vec3 localPos;

layout (binding = 0) uniform samplerCube environmentMap;

#define SAMPLE_DELTA 0.01

#define PI 3.14159265359
#define TWO_PI 2 * PI
#define HALF_PI 0.5 * PI

//#define PHI_SAMPLES floor(TWO_PI / SAMPLE_DELTA)
//#define THETA_SAMPLES floor(HALF_PI / SAMPLE_DELTA)
#define INV_TOTAL_SAMPLES 1 / ( floor(TWO_PI / SAMPLE_DELTA) * floor(HALF_PI / SAMPLE_DELTA) ) 

void main()
{
    const vec3 normal = normalize(localPos);
    const vec3 right = cross(vec3(0.0, 1.0, 0.0), normal);
    const vec3 up    = cross(normal, right);

    vec3 irradiance = vec3(0.0);
    for(float phi = 0.0; phi < TWO_PI; phi += SAMPLE_DELTA)
    {
        const float sinPhi = sin(phi);
        const float cosPhi = cos(phi);
        
        for(float theta = 0.0; theta < HALF_PI; theta += SAMPLE_DELTA)
        {
            const float sinTheta = sin(theta);
            const float cosTheta = cos(theta);

            vec3 tangentSample = vec3(sinTheta * cosPhi,  sinTheta * sinPhi, cosTheta);

            vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * normal; 

            irradiance += texture(environmentMap, sampleVec).rgb * cosTheta * sinTheta;
        }
    }
    irradiance = PI * irradiance * INV_TOTAL_SAMPLES;

    FragColor = vec4(irradiance, 1.0);
}