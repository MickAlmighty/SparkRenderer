#include "CUDA/kernel.cuh"

#include <device_launch_parameters.h>

#include "BinaryHeap.cuh"
#include "DeviceMemory.h"
#include "DeviceTimer.cuh"
#include "Map.cuh"
#include "MemoryAllocator.cuh"
#include "MemoryManager.cuh"
#include "Node.cuh"
#include "Timer.h"


namespace spark {
	namespace cuda {
		__device__ Map* map = nullptr;
		__device__ MemoryAllocator* allocator[128];

		__host__ std::vector<unsigned int> runKernel(int blocks, int threads, int* path, unsigned int* agentPaths, void* kernelMemory, const Map& mapHost)
		{
			{
				cudaDeviceSynchronize();
				//Timer t3("	findPath");
				findPath << <blocks, threads, mapHost.width * mapHost.height * sizeof(unsigned int) + 32 * 8 * sizeof(Node) >> > (path, agentPaths, kernelMemory);
				cudaDeviceSynchronize();
				gpuErrchk(cudaGetLastError());
			}

			const size_t agentPathSize = mapHost.width * mapHost.height * 2;
			const size_t agentPathsBufferSize = agentPathSize * blocks * 32;
			std::vector<unsigned int> paths(agentPathsBufferSize);
			cudaMemcpy(paths.data(), agentPaths, agentPathsBufferSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			return paths;
		}

		__host__ void initMap(float* nodes, int width, int height)
		{
			PROFILE_FUNCTION();
			createMap << <1, 1 >> > (nodes, width, height);
			gpuErrchk(cudaGetLastError());
		}

		__global__  void createMap(float* nodes, int width, int height)
		{
			if (map != nullptr)
			{
				delete map;
			}
			map = new Map(width, height, nodes);
		}

		__global__ void findPath(int* path, unsigned int* agentPaths, void* kernelMemory)
		{
			extern __shared__ unsigned char sharedMemory[];

			unsigned int* closedNodesLookup = reinterpret_cast<unsigned int*>(sharedMemory);

			if (threadIdx.x == 0)
			{
				memset(sharedMemory, 0, sizeof(unsigned int) * map->width * map->height + 8 * 32 * sizeof(Node));
				//allocator[blockIdx.x] = new MemoryAllocator(sizeof(Node) * map->width * map->height * 8 * 32);
			}
			__syncthreads();

			Node* neighborsMemoryPool = reinterpret_cast<Node*>(sharedMemory + map->width * map->height * sizeof(unsigned int));
			Node* neighbors = neighborsMemoryPool + threadIdx.x * 8; // 8 is the neighbor array size

			const size_t agentPathMemorySize = map->width * map->height * 2;
			const size_t agentPathWarpMemorySize = agentPathMemorySize * 32;
			const int agentPathOffset = agentPathWarpMemorySize * blockIdx.x + agentPathMemorySize * threadIdx.x;
			unsigned int* agentPath = agentPaths + agentPathOffset;

			const size_t memoryOffset = map->width * map->height * 32;
			const size_t threadMemOffset = memoryOffset * threadIdx.x;
			const size_t blockSizeElements = memoryOffset * 16;
			const size_t blockMemOffset = blockSizeElements * blockIdx.x;
			BinaryHeap<Node> heap(static_cast<Node*>(kernelMemory) + blockMemOffset + threadMemOffset);
			MemoryManager manager = MemoryManager(static_cast<Node*>(kernelMemory) + blockMemOffset + threadMemOffset + map->width * map->height * 8);

			const unsigned int startEndPointsBlockMemorySize = 4 * 32;
			int startPoint[] = { path[startEndPointsBlockMemorySize * blockIdx.x + 4 * threadIdx.x + 0], path[startEndPointsBlockMemorySize * blockIdx.x + 4 * threadIdx.x + 1] };
			int endPoint[] = { path[startEndPointsBlockMemorySize * blockIdx.x + 4 * threadIdx.x + 2], path[startEndPointsBlockMemorySize * blockIdx.x + 4 * threadIdx.x + 3] };

			if (startPoint[0] == endPoint[0] && startPoint[1] == endPoint[1])
				return;

			Node* finishNode = nullptr;

			const Node startNode(startPoint, 0.0f);
			heap.insert(startNode);

			int whileLoopCounter = 0;
			while (!finishNode)
			{
				++whileLoopCounter;
				if (heap.size == 0)
				{
					printf("Heap is empty\n");
					break;
				}

				if (heap.size >= map->width * map->height * 8)
				{
					printf("Heap size = %llu limit reached!\n", heap.size);
					break;
				}

				if (whileLoopCounter >= map->width * map->height * 8)
				{
					printf("Closed nodes limit reached!\n");
					break;
				}

				auto theBestNode = heap.pop_front();

				const auto closedNode = manager.allocate<Node>(theBestNode); //it will be deleted with allocator deletion

				const int nodeIndex = map->getTerrainNodeIndex(closedNode->pos[0], closedNode->pos[1]);
				atomicOr(closedNodesLookup + nodeIndex, 1 << threadIdx.x);

				closedNode->getNeighbors(map, neighbors);

				int i = 0;
				//nvstd::function<bool(const Node& node)> findOpened = [&neighbors, &i] __device__(const Node& node)
				//{
				//	if (node.pos[0] != neighbors[i].pos[0] ||
				//		node.pos[1] != neighbors[i].pos[1])
				//	{
				//		return false;
				//	}
				//	
				//	//betterFunctionG
				//	return neighbors[i].distanceFromBeginning < node.distanceFromBeginning;
				//};
#pragma unroll
				for (i = 0; i < 8; ++i)
				{
					if (!neighbors[i].valid)
					{
						continue;
					}

					const int nodeIdx = map->getTerrainNodeIndex(neighbors[i].pos[0], neighbors[i].pos[1]);
					if (closedNodesLookup[nodeIdx] & (1 << threadIdx.x))
					{
						continue;
					}

					neighbors[i].parent = closedNode;

					if (neighbors[i].pos[0] == endPoint[0] &&
						neighbors[i].pos[1] == endPoint[1])
					{
						finishNode = manager.allocate<Node>(neighbors[i]);
						break;
					}

					neighbors[i].valueH = neighbors[i].measureManhattanDistance(endPoint);
					const float functionG = neighbors[i].distanceFromBeginning;
					const int neighborPos[2] = { neighbors[i].pos[0], neighbors[i].pos[1] };
					const float terrainValue = map->getTerrainValue(neighborPos[0], neighborPos[1]);
					neighbors[i].valueF = (1.0f - terrainValue) * (neighbors[i].valueH + functionG);

					/*const int nodeToSwapIndex = heap.findIndex_if(findOpened);

					if (nodeToSwapIndex != -1)
					{
						heap.removeValue(nodeToSwapIndex);
						heap.insert(neighbors[i]);
					}
					else*/
					{
						heap.insert(neighbors[i]);
					}
				}
			}

			if (!finishNode)
			{
				return;
			}

			int pathLength = 0;
			finishNode->getPathLength(pathLength);
			agentPath[0] = pathLength;
			finishNode->recreatePath(agentPath + 1, pathLength);
			//printf("thread %d, block %d, path length %d\nopened nodes %d, loopCounter %d\n", threadIdx.x, blockIdx.x, pathLength, int(heap.size), whileLoopCounter);
			printf("GPU: Nodes processed %d, nodesToProcess %d, pathSize %d\n", whileLoopCounter, int(heap.size), pathLength);
		}

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
	}
}

