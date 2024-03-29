#type compute
#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(r16f, binding = 0) uniform image2D output_image;

#define HISTOGRAM_BINS 256

layout (std430, binding = 0) buffer LuminanceHistogram {
    uint luminanceHistogram[HISTOGRAM_BINS];
};

layout (push_constant) uniform PushConstants
{
    uint pixelCount;
    float minLogLuminance;
    float logLuminanceRange;
    float deltaTime;
    float tau;
} u_Uniforms;

shared float histogramShared[HISTOGRAM_BINS];

void main()
{
    float countForThisBin = float(luminanceHistogram[gl_LocalInvocationIndex]);
    histogramShared[gl_LocalInvocationIndex] = countForThisBin * float(gl_LocalInvocationIndex);

    barrier();

    for (uint histogramSampleIndex = (HISTOGRAM_BINS >> 1); histogramSampleIndex > 0; histogramSampleIndex>>=1)
    {
        if (gl_LocalInvocationIndex < histogramSampleIndex)
        {
            histogramShared[gl_LocalInvocationIndex] += histogramShared[gl_LocalInvocationIndex + histogramSampleIndex];
        }

        barrier();
    }

    if (gl_LocalInvocationIndex == 0)
    {
        float weightedLogAverage = (histogramShared[0].x / max(float(u_Uniforms.pixelCount) - countForThisBin, 1.0f)) - 1.0f;
        float weightedAverageLuminance = exp2(((weightedLogAverage / 254.0) * u_Uniforms.logLuminanceRange) + u_Uniforms.minLogLuminance);
        float luminanceLastFrame = imageLoad(output_image, ivec2(0,0)).x;
        float adaptedLuminance = luminanceLastFrame + (weightedAverageLuminance - luminanceLastFrame) * (1 - exp(-u_Uniforms.deltaTime * u_Uniforms.tau));
        imageStore(output_image, ivec2(0.0), vec4(adaptedLuminance, 0, 0, 0));
    }
}