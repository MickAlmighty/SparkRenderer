#include "CUDA/kernel.cuh"

#include <deque>

#include <device_launch_parameters.h>

#include "Agent.cuh"
#include "BinaryHeap.cuh"
#include "DeviceMemory.h"
#include "DeviceTimer.cuh"
#include "List.cuh"
#include "Map.cuh"
#include "MemoryAllocator.cuh"
#include "MemoryManager.cuh"
#include "Node.cuh"
#include "Timer.h"
#include <glm/vec2.hpp>

namespace spark {
	namespace cuda {
		__device__ Map* map = nullptr;
		__device__ MemoryAllocator* allocator[32];

		__host__ void runKernel(int* path, unsigned int* agentPaths)
		{
			{
				cudaDeviceSynchronize();
				Timer t3("	findPath");
				findPath << <32, 32, 400 * sizeof(unsigned int) + 32 * 8 * sizeof(Node) >> > (path, agentPaths);
				cudaDeviceSynchronize();
				gpuErrchk(cudaGetLastError());
			}

			const size_t agentPathSize = 400 * 2;
			const size_t agentPathsBufferSize = agentPathSize * 1024;
			std::vector<unsigned int> paths(agentPathsBufferSize);
			cudaMemcpy(paths.data(), agentPaths, agentPathsBufferSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			std::vector<std::vector<glm::ivec2>> pathsForAgents;
			pathsForAgents.reserve(1024);
			for (int i = 0; i < 1024; ++i)
			{
				const size_t pathSize = paths[agentPathSize * i];
				pathsForAgents.push_back(std::vector<glm::ivec2>(pathSize));
				memcpy(pathsForAgents[i].data(), reinterpret_cast<int*>(paths.data()) + agentPathSize * i + 1, sizeof(int) *  pathSize * 2);
			}
			float f = 5.0f;
		}

		__host__ void initMap(float* nodes, int width, int height)
		{
			Timer t("		initMap");
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

		__global__ void findPath(int* path, unsigned int* agentPaths)
		{
			extern __shared__ unsigned char sharedMemory[];
			
			unsigned int* closedNodesLookup = reinterpret_cast<unsigned int*>(sharedMemory);

			if (threadIdx.x == 0)
			{
				memset(sharedMemory, 0, sizeof(unsigned int) * map->width * map->height + 8 * 32 * sizeof(Node));
				allocator[blockIdx.x] = new MemoryAllocator(sizeof(Node) * map->width * map->height * 8 * 32);
			}
			__syncthreads();

			Node* neighborsMemoryPool = reinterpret_cast<Node*>(sharedMemory + 400 * sizeof(unsigned int));
			Node* neighbors = neighborsMemoryPool + threadIdx.x * 8; // 8 is the neighbor array size

			const size_t memoryOffset = map->width * map->height * 8;
			const size_t kernelMemOffset = memoryOffset * threadIdx.x;

			const size_t agentPathMemorySize = map->width * map->height * 2;
			const size_t agentPathWarpMemorySize = agentPathMemorySize * 32;
			const int agentPathOffset = agentPathWarpMemorySize * blockIdx.x + agentPathMemorySize * threadIdx.x;
			unsigned int* agentPath = agentPaths + agentPathOffset;

			BinaryHeap<Node> heap(allocator[blockIdx.x]->ptr<Node>(kernelMemOffset));
			MemoryManager manager = MemoryManager(allocator[blockIdx.x]->ptr<Node>(kernelMemOffset + map->width * map->height * 7));
			/*int startPoint[] = { *(path + 4 * threadIdx.x + 0), *(path + 4 * threadIdx.x + 1) };
			int endPoint[] = { *(path + 4 * threadIdx.x + 2), *(path + 4* threadIdx.x + 3) };*/
			DeviceTimer timer;
			int startPoint[] = { *(path + 0), *(path + 1) };
			int endPoint[] = { *(path + 2), *(path + 3) };

			Node* finishNode = nullptr;

			const Node startNode(startPoint, 0.0f);
			heap.insert(startNode);

			int whileLoopCounter = 0;
			while (true)
			{
				++whileLoopCounter;
				if (heap.size == 0)
				{
					break;
				}

				auto theBestNode = heap.pop_front();

				const auto closedNode = manager.allocate<Node>(theBestNode); //it will be deleted with allocator deletion

				if (closedNode->pos[0] == endPoint[0] &&
					closedNode->pos[1] == endPoint[1])
				{
					finishNode = closedNode;
					break;
				}

				const int nodeIndex = map->getTerrainNodeIndex(closedNode->pos[0], closedNode->pos[1]);
				atomicOr(closedNodesLookup + nodeIndex, 1 << threadIdx.x);

				closedNode->getNeighbors(map, neighbors);

				int i = 0;
				/*nvstd::function<bool(const Node& node)> findOpened = [&neighbors, &i] __device__(const Node& node)
				{
					const bool positionEqual = node.pos[0] == neighbors[i].pos[0] &&
						node.pos[1] == neighbors[i].pos[1];

					if (!positionEqual)
					{
						return false;
					}

					const bool betterFunctionG = neighbors[i].distanceFromBeginning < node.distanceFromBeginning;

					return betterFunctionG;
				};*/

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

					neighbors[i].valueH = neighbors[i].measureManhattanDistance(endPoint);
					const float functionG = neighbors[i].distanceFromBeginning;
					const int neighborPos[2] = { neighbors[i].pos[0], neighbors[i].pos[1] };
					const float terrainValue = map->getTerrainValue(neighborPos[0], neighborPos[1]);
					neighbors[i].valueF = (1.0f - terrainValue) * (neighbors[i].valueH + functionG);

					//const int nodeToSwapIndex = heap.findIndex_if(findOpened);

					//if (nodeToSwapIndex != -1)
					//{
					//	//timer.reset();
					//	heap.removeValue(nodeToSwapIndex);
					//	heap.insert(neighbors[i]);
					//	//timer.printTime("	Node insertion after node deletion %f ms\n");
					//}
					//else
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
			
			__syncthreads();
			if (threadIdx.x == 0)
			{
				delete allocator[blockIdx.x];
			}
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

