#include "CUDA/kernel.cuh"

#include <cstdlib>
#include <iostream>

struct Pixel
{
	unsigned char r = 0;
	unsigned char g = 0;
	unsigned char b = 0;
};

__global__ void changeColors(Pixel* pixel_dev, int brightness)
{
	//pixel_dev[blockIdx.x].r = 166;
	//int index = blockIdx.x + threadIdx.x;
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;
	int r = pixel_dev[offset].r + brightness;
	int g = pixel_dev[offset].g + brightness;
	int b = pixel_dev[offset].b + brightness;

	if (r < 0)
		r = 0;
	if (g < 0)
		g = 0;
	if (b < 0)
		b = 0;

	if (r > 255)
		r = 255;
	if (g > 255)
		g = 255;
	if (b > 255)
		b = 255;

	pixel_dev[offset].r = r;
	pixel_dev[offset].g = g;
	pixel_dev[offset].b = b;
}

__global__ void horizontalGaussianBlur(Pixel* pixel_dev)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;
	static const float weight[5] = { 0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162 };
	int colorCanal = 0;

	colorCanal = pixel_dev[offset].r * weight[0];
	for (int i = 1; i < 5; i++)
	{
		colorCanal += pixel_dev[offset + i].r * weight[i];
		colorCanal += pixel_dev[offset - i].r * weight[i];
	}
	pixel_dev[offset].r = colorCanal;

	colorCanal = pixel_dev[offset].g * weight[0];
	for (int i = 1; i < 5; i++)
	{
		colorCanal += pixel_dev[offset + i].g * weight[i];
		colorCanal += pixel_dev[offset - i].g * weight[i];
	}
	pixel_dev[offset].g = colorCanal;


	colorCanal = pixel_dev[offset].b * weight[0];
	for (int i = 1; i < 5; i++)
	{
		colorCanal += pixel_dev[offset + i].b * weight[i];
		colorCanal += pixel_dev[offset - i].b * weight[i];
	}
	pixel_dev[offset].b = colorCanal;
}

__global__ void verticalGaussianBlur(Pixel* pixel_dev)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;
	//static int executed = 0;
	static const float weight[5] = { 0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162 };
	int colorCanal = 0;
	//printf("\n Block %d Offset %d Executed %d", blockIdx.x, offset, executed);
	colorCanal = pixel_dev[offset].r * weight[0];
	for (int i = 1; i < 5; i++)
	{
		int index = offset + i * gridDim.x;
		if (index < gridDim.x * gridDim.y)
		{
			colorCanal += pixel_dev[offset + i * gridDim.x].r * weight[i];
		}
		index = offset - i * gridDim.x;
		if (index >= 0)
		{
			colorCanal += pixel_dev[offset - i * gridDim.x].r * weight[i];
		}
	}
	pixel_dev[offset].r = colorCanal;

	colorCanal = pixel_dev[offset].g * weight[0];
	for (int i = 1; i < 5; i++)
	{
		int index = offset + i * gridDim.x;
		if (index < gridDim.x * gridDim.y)
		{
			colorCanal += pixel_dev[offset + i * gridDim.x].g * weight[i];
		}
		index = offset - i * gridDim.x;
		if (index >= 0)
		{
			colorCanal += pixel_dev[offset - i * gridDim.x].g * weight[i];
		}
	}
	pixel_dev[offset].g = colorCanal;


	colorCanal = pixel_dev[offset].b * weight[0];
	for (int i = 1; i < 5; i++)
	{
		int index = offset + i * gridDim.x;
		if (index < gridDim.x * gridDim.y)
		{
			colorCanal += pixel_dev[offset + i * gridDim.x].b * weight[i];
		}
		index = offset - i * gridDim.x;
		if (index >= 0)
		{
			colorCanal += pixel_dev[offset - i * gridDim.x].b * weight[i];
		}
	}
	//executed++;
	pixel_dev[offset].b = colorCanal;

}

__global__ void sampleAddition(float* data)
{
	data[threadIdx.x] = threadIdx.x;
	//vector.data();
}

__host__ void runKernel()
{

	float* data, *cudaData;
	data = (float*)malloc(10 * sizeof(float));
	cudaMalloc(&cudaData, 10 * sizeof(float));

	sampleAddition << <1, 10 >> > (data);

	cudaMemcpy(data, cudaData, 10 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(cudaData);

	for (int i = 0; i < 10; ++i)
	{
		std::cout << data[i] << std::endl;
	}
}

void runKernel(int iterations)
{
	/*std::clock_t start;
	double duration;
	start = std::clock();

	ImageLoader* imgLoader = ImageLoader::getInstance();
	int width = imgLoader->width, height = imgLoader->height, channels = imgLoader->channels;
	int size = width * height;*/
	Pixel *pixels = static_cast<Pixel*>(malloc(1024 * 1024 * sizeof(Pixel)));

	int size = 1024 * 1024;
	Pixel *pixels_dev; 

	cudaMalloc(&pixels_dev, size * sizeof(Pixel));
	cudaMemcpy(pixels_dev, pixels, size * sizeof(Pixel), cudaMemcpyHostToDevice);
	
	dim3 grid(1024, 1024);

	for (int i = 0; i < iterations; i++)
	{
		horizontalGaussianBlur <<< grid, 1 >> > (pixels_dev);
		verticalGaussianBlur <<< grid, 1 >> > (pixels_dev);
		changeColors <<< grid, 1 >> > (pixels_dev, 12);
		cudaGetLastError();
	}

	free(pixels);
}
