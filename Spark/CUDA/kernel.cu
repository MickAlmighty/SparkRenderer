#include "CUDA/kernel.cuh"

#include <cstdlib>
#include <iostream>

__global__ void checkMapValues(Map* mapDev)
{
	int index = 0;
	for(int i = 0; i < mapDev->width; ++i)
	{
		for(int j = 0; j < mapDev->height; ++j)
		{
			int resultIndex = mapDev->getTerrainNodeIndex(i, j);
			if (resultIndex == index)
			{
				if (mapDev->nodes[resultIndex] == 1.0f)
				{
					float f = mapDev->getTerrainNodeIndex(i, j);
				}
			}
			++index;
		}
	}
}

__global__  void attachNodes(Map* map, float* nodes)
{
	map->nodes = nodes;
}

void runKernel(Map* map, float* nodes)
{
	attachNodes<<<1, 1>>>(map, nodes);
	cudaDeviceSynchronize();
	checkMapValues<<<1,1>>>(map);
	cudaDeviceSynchronize();
}