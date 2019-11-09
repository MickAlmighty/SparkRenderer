#include "CUDA/kernel.cuh"

#include <cstdlib>
#include <iostream>

#include "Node.cuh"

namespace spark {
	namespace cuda {

		__global__ void checkMapValues(Map* mapDev)
		{
			int index = 0;
			for (int i = 0; i < mapDev->width; ++i)
			{
				for (int j = 0; j < mapDev->height; ++j)
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

		__global__ void findPath(Map* map, int* path)
		{
			int startPoint[] = { path[0], path[1] };
			int endPoint[] = { path[2], path[3] };

			Node startNode(endPoint, 0.0f);

			for(int i = 0; i < 10000; ++i)
			{
				Node* nodes = startNode.getNeighbors(map);
				delete[] nodes;
			}

		//#todo: add first node to open list
			while(true)
			{
			//#todo: get first node from openedList

			//#todo: check if node position is endPos
				if(startNode.pos[0] == endPoint[0] && 
					startNode.pos[1] == endPoint[1])
				{
					break;
				}
			}
		}

		void runKernel(Map* map, float* nodes, int* path)
		{
			attachNodes << <1, 1 >> > (map, nodes);
			cudaDeviceSynchronize();
			checkMapValues << <1, 1 >> > (map);
			findPath << <1, 1 >> > (map, path);
			cudaDeviceSynchronize();
		}
	}
}