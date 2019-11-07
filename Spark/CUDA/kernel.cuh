#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct Pixel;
__global__ void changeColors(Pixel* pixel_dev, int brightness);
__global__ void horizontalGaussianBlur(Pixel* pixel_dev);
__global__ void verticalGaussianBlur(Pixel* pixel_dev);
__global__ void sampleAddition(float* data);
__host__ void runKernel();
void runKernel(int iterations);