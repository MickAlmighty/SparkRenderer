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

		__host__ void runKernel(int* path, int* memSize, Agent* agents)
		{
			{
				cudaDeviceSynchronize();
				Timer t3("	findPath");
				findPath << <32, 32, 400 * sizeof(unsigned int) + 32 * 8 * sizeof(Node) >> > (path, memSize, agents);
				cudaDeviceSynchronize();
				gpuErrchk(cudaGetLastError());
			}
			
			Timer t("	CUDA KERNEL: filling pathBuffer");
			//cudaDeviceSynchronize();
			int memorySize = 0;
			cudaMemcpy(&memorySize, memSize, sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			const DeviceMemory<int> pathBuffer = DeviceMemory<int>::AllocateElements(memorySize);
			fillPathBuffer<<<1, 1>>>(agents, pathBuffer.ptr, 1024);
			gpuErrchk(cudaGetLastError());

			{
				Timer t4("	CUDA KERNEL: deleting deleteAgentPaths");
				cudaDeviceSynchronize();
				deleteAgentPaths << <32, 32 >> > (agents);
				gpuErrchk(cudaGetLastError());
			}
			
			Timer t2("	CUDA KERNEL: Filling agent paths");
			std::vector<Agent> agentLookup(1024);
			cudaMemcpy(agentLookup.data(), agents, sizeof(Agent) * 1024, cudaMemcpyDeviceToHost);

			std::vector<glm::ivec2> paths(memorySize / 2);
			cudaMemcpy(paths.data(), pathBuffer.ptr, sizeof(int) * memorySize, cudaMemcpyDeviceToHost);
			
			//cudaDeviceSynchronize();
			std::vector<std::vector<glm::ivec2>> agentPaths;
			agentPaths.reserve(1024);
			for (int i = 0; i < 1024; ++i)
			{
				agentPaths.push_back(std::vector<glm::ivec2>(agentLookup[i].pathSize));
				memcpy(agentPaths[i].data(), reinterpret_cast<int*>(paths.data()) + agentLookup[i].indexBegin, sizeof(int) * agentLookup[i].pathSize * 2);
				/*for (int j = agentLookup[i].indexBegin; j < agentLookup[i].pathSize * 2; j += 2)
				{
					agentPath.push_back({ false, {paths[j], paths[j + 1]} });
				}*/
			}

			/*for(const glm::ivec2& wayPoint : agentPaths[0])
			{
				std::cout << "{" << wayPoint.x << " " << wayPoint.y << "} ";
			}
			std::cout << "\n";*/
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
			map = new Map();
			map->nodes = new float[width * height];
			memcpy(map->nodes, nodes, width * height * sizeof(float));
			//map->nodes = nodes;
			map->width = width;
			map->height = height;
		}

		__global__ void findPath(int* path, int* memSize, Agent* agents)
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
			Node* neighbors = neighborsMemoryPool + threadIdx.x * 8; // 8 is neighbor array size

			const size_t memoryOffset = map->width * map->height * 8;
			const size_t kernelMemOffset = memoryOffset * threadIdx.x;
			MemoryManager manager = MemoryManager(allocator[blockIdx.x]->ptr<Node>(kernelMemOffset + map->width * map->height * 7));
			/*int startPoint[] = { *(path + 4 * threadIdx.x + 0), *(path + 4 * threadIdx.x + 1) };
			int endPoint[] = { *(path + 4 * threadIdx.x + 2), *(path + 4* threadIdx.x + 3) };*/
			DeviceTimer timer;
			int startPoint[] = { *(path + 0), *(path + 1) };
			int endPoint[] = { *(path + 2), *(path + 3) };
			
			const Node startNode(startPoint, 0.0f);
			BinaryHeap<Node> heap(allocator[blockIdx.x]->ptr<Node>(kernelMemOffset));
			Node* finishNode = nullptr;
			Iterator<Node>* closedNodeIt = manager.allocate<Iterator<Node>>(startNode);
			heap.insert(startNode);
			int whileLoopCounter = 0;
			while (true)
			{
				++whileLoopCounter;
				if (heap.size == 0)
				{
					break;
				}

				//timer.reset();
				auto closedNode = heap.pop_front();
				//timer.printTime("Pop front from heap %f ms\n");

				//timer.reset();
				closedNodeIt->placeNext(manager.allocate<Iterator<Node>>(closedNode));
				closedNodeIt = closedNodeIt->next;
				//timer.printTime("Inserting closed node to closed list %f ms\n");

				if (closedNodeIt->value.pos[0] == endPoint[0] &&
					closedNodeIt->value.pos[1] == endPoint[1])
				{
					finishNode = &closedNodeIt->value;
					break;
				}
				//timer.printTime("Checking if the node is finished node %f ms\n");

				//timer.reset();
				const int nodeIndex = map->getTerrainNodeIndex(closedNode.pos[0], closedNode.pos[1]);
				atomicOr(closedNodesLookup + nodeIndex, 1 << threadIdx.x);
				//timer.printTime("Bitwise setting that node is now closed %f ms\n");

				//timer.reset();
				//Node neighbors[8];
				//memset(neighbors, 0, 8 * sizeof(Node));
				closedNode.getNeighbors(map, neighbors);
				//timer.printTime("Getting node neighbors %f ms\n");

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

					//timer.reset();
					const int nodeIdx = map->getTerrainNodeIndex(neighbors[i].pos[0], neighbors[i].pos[1]);
					if (closedNodesLookup[nodeIdx] & (1 << threadIdx.x))
					{
						continue;
					}
					//timer.printTime("	Check if node is closed %f ms\n");

					//timer.reset();
					neighbors[i].parent = &closedNodeIt->value;

					neighbors[i].valueH = neighbors[i].measureManhattanDistance(endPoint);
					const float functionG = neighbors[i].distanceFromBeginning;
					const int neighborPos[2] = { neighbors[i].pos[0], neighbors[i].pos[1] };
					const float terrainValue = map->getTerrainValue(neighborPos[0], neighborPos[1]);
					neighbors[i].valueF = (1.0f - terrainValue) * (neighbors[i].valueH + functionG);
					//timer.printTime("	Heuristic calculation %f ms\n");

					//timer.reset();
					//const int nodeToSwapIndex = heap.findIndex_if(findOpened);
					////timer.printTime("	Find better opened node %f ms\n");

					//if (nodeToSwapIndex != -1)
					//{
					//	//timer.reset();
					//	heap.removeValue(nodeToSwapIndex);
					//	heap.insert(neighbors[i]);
					//	//timer.printTime("	Node insertion after node deletion %f ms\n");
					//}
					//else
					{
						//timer.reset();
						heap.insert(neighbors[i]);
						//timer.printTime("	Node insertion %f ms\n");
					}
				}
			}

			if (!finishNode)
			{
				return;
			}

			int pathLength = 0;
			finishNode->getPathLength(pathLength);
			atomicAdd(memSize, pathLength * 2);
			int* agentPath = static_cast<int*>(malloc(sizeof(int) * pathLength * 2));
			finishNode->recreatePath(agentPath, pathLength);
			//printf("thread %d, block %d, path length %d\nopened nodes %d, loopCounter %d\n", threadIdx.x, blockIdx.x, pathLength, int(heap.size), whileLoopCounter);

			agents[blockIdx.x * 32 + threadIdx.x].pathOutput = agentPath;
			agents[blockIdx.x * 32 + threadIdx.x].pathSize = pathLength;
			
			__syncthreads();
			if (threadIdx.x == 0)
			{
				delete allocator[blockIdx.x];
			}
			//timer2.printTime("Kernel overall time %f ms\n");
		}

		__global__ void fillPathBuffer(Agent* agents, int* pathBuffer, int numberOfAgents)
		{
			int pathIndex = 0;
			for (int i = 0; i < numberOfAgents; ++i)
			{
				agents[i].indexBegin = pathIndex;
				/*for (int j = 0; j < agents[i].pathSize * 2; ++j)
				{
					pathBuffer[pathIndex] = agents[i].pathOutput[j];
					++pathIndex;
				}*/
				const size_t numberOfBytes = sizeof(int) * agents[i].pathSize * 2;
				memcpy(pathBuffer + agents[i].indexBegin, agents[i].pathOutput, numberOfBytes);
				pathIndex += agents[i].pathSize * 2;
			}
		}

		__global__ void deleteAgentPaths(Agent* agents)
		{
			free(agents[blockIdx.x * 32 + threadIdx.x].pathOutput);
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

