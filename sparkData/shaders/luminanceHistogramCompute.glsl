#type compute
#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = 0) uniform image2D img_input;

#define HISTOGRAM_BINS 256
#define EPSILON 0.005

layout (location = 0) uniform ivec2 inputTextureSize;
layout (location = 1) uniform float minLogLuminance;
layout (location = 2) uniform float oneOverLogLuminanceRange;

layout (std430, binding = 0) buffer LuminanceHistogram {
    uint luminanceHistogram[HISTOGRAM_BINS];
};

shared uint histogramShared[HISTOGRAM_BINS];

float getLuminance(vec3 color);
uint HDRToHistogramBin(vec3 hdrColor);

void main() 
{
    histogramShared[gl_LocalInvocationIndex] = 0; // gl_LocalInvocationIndex from 0 to 16 * 16 - 1

    barrier();

    if (gl_GlobalInvocationID.x < inputTextureSize.x || gl_GlobalInvocationID.y < inputTextureSize.y)
    {
        ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
        vec4 pixColor = imageLoad(img_input, pixel_coords);
        uint binIndex = HDRToHistogramBin(pixColor.xyz);
        atomicAdd(histogramShared[binIndex], 1);
    }

    barrier();

    atomicAdd(luminanceHistogram[gl_LocalInvocationIndex], histogramShared[gl_LocalInvocationIndex]);
}

float getLuminance(vec3 color)
{
    return dot(color, vec3(0.2127f, 0.7152f, 0.0722f));
}

uint HDRToHistogramBin(vec3 hdrColor)
{
    float luminance = getLuminance(hdrColor);

    if(luminance < EPSILON)
    {
        return 0;
    }

    float logLuminance = clamp((log2(luminance) - minLogLuminance) * oneOverLogLuminanceRange, 0.0f, 1.0f);
    return uint(logLuminance * 254.0 + 1.0);
}