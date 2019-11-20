#include "CUDA/kernel.cuh"

#include <deque>

#include <device_launch_parameters.h>

#include "Agent.cuh"
#include "DeviceTimer.cuh"
#include "List.cuh"
#include "Map.cuh"
#include "MemoryAllocator.cuh"
#include "Node.cuh"
#include "Timer.h"

namespace spark {
	namespace cuda {
		__device__ Map* map = nullptr;

		__host__ void runKernel(int* path, int* memSize, Agent* agents)
		{

			//checkMapValues <<<1, 1 >>> (map);
			cudaDeviceSynchronize();
			Timer t3("	findPath");
			findPath << <1, 1, 400 * sizeof(unsigned int) >> > (path, memSize, agents);
			gpuErrchk(cudaGetLastError());
			cudaDeviceSynchronize();
			//int memorySize = 0;
			//int* pathBuffer = nullptr;
			//{
			//	Timer t("CUDA KERNEL: filling pathBuffer");
			//	
			//	cudaMemcpy(&memorySize, memSize, sizeof(int), cudaMemcpyDeviceToHost);
			//	
			//	cudaMalloc(&pathBuffer, sizeof(int) * memorySize);
			//	//cudaDeviceSynchronize();
			//	fillPathBuffer <<<1, 1>>> (agents, pathBuffer, 1);
			//	cudaDeviceSynchronize();
			//}
			//
			//{
			//	Timer t2("CUDA KERNEL: Filling agent path");
			//	Agent* agentLookup = new Agent[1];
			//	cudaMemcpy(agentLookup, agents, sizeof(Agent), cudaMemcpyDeviceToHost);
			//	int* paths = new int[memorySize];
			//	cudaMemcpy(paths, pathBuffer, sizeof(int) * memorySize, cudaMemcpyDeviceToHost);

			//	for (int i = 0; i < 1; ++i)
			//	{
			//		std::deque<std::pair<bool, glm::ivec2>> agentPath;
			//		for (int j = agentLookup[i].indexBegin; j < agentLookup[i].pathSize * 2; j += 2)
			//		{
			//			agentPath.push_back({ false, {paths[j], paths[j + 1]} });
			//		}
			//	}

			//	delete[] paths;
			//	cudaFree(pathBuffer);
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
			extern __shared__ int closedNodesLookup[];
			/*int startPoint[] = { *(path + 4 * threadIdx.x + 0), *(path + 4 * threadIdx.x + 1) };
			int endPoint[] = { *(path + 4 * threadIdx.x + 2), *(path + 4* threadIdx.x + 3) };*/
			DeviceTimer timer2;
			int startPoint[] = { *(path + 0), *(path + 1) };
			int endPoint[] = { *(path + 2), *(path + 3) };
			DeviceTimer timer;
			MemoryAllocator allocator = MemoryAllocator(sizeof(Node) * 400 * 8);
			timer.printTime("Memory allocation %f ms\n");

			const Node startNode(startPoint, 0.0f);
			List<Node> openNodes(&allocator);
			List<Node> closedNodes(&allocator);
			Node* finishNode = nullptr;
			openNodes.insert(startNode);

			while (true)
			{
				DeviceTimer loopTimer;
				if (openNodes.size == 0)
				{
					break;
				}

				const auto closedNode = openNodes.pop_front();
				closedNodes.insert(closedNode);

				//timer.reset();
				unsigned int beforeChange = closedNodesLookup[map->getTerrainNodeIndex(closedNode->value.pos[0], closedNode->value.pos[1])];
				const unsigned int change = beforeChange | (1 << threadIdx.x);
				closedNodesLookup[map->getTerrainNodeIndex(closedNode->value.pos[0], closedNode->value.pos[1])] = beforeChange | change;
				//timer.printTime("Bitwise setting that node is now closed %f ms\n");

				if (closedNode->value.pos[0] == endPoint[0] &&
					closedNode->value.pos[1] == endPoint[1])
				{
					finishNode = &closedNode->value;
					break;
				}

				timer.reset();
				Node neighbors[8];
				closedNode->value.getNeighbors(map, neighbors);
				timer.printTime("Getting node neighbors %f ms\n");

				for (int i = 0; i < 8; ++i)
				{
					/*nvstd::function<bool(const Node& node)> f = [neighborIt] __device__ (const Node& node)
					{
						return node.pos[0] == neighborIt->value.pos[0] &&
							node.pos[1] == neighborIt->value.pos[1];
					};*/
					if (!neighbors[i].valid)
					{
						continue;
					}

					if (closedNodesLookup[map->getTerrainNodeIndex(neighbors[i].pos[0], neighbors[i].pos[1])] & (1 << threadIdx.x))
					{
						continue;
					}

					timer.reset();
					neighbors[i].parent = &closedNode->value;

					neighbors[i].valueH = neighbors[i].measureManhattanDistance(endPoint);
					const float functionG = neighbors[i].distanceFromBeginning;
					const int neighborPos[2] = { neighbors[i].pos[0], neighbors[i].pos[1] };
					const float terrainValue = map->getTerrainValue(neighborPos[0], neighborPos[1]);
					neighbors[i].valueF = (1.0f - terrainValue) * (neighbors[i].valueH + functionG);
					timer.printTime("	Heuristic calculation %f ms\n");

					nvstd::function<bool(const Node& node)> findOpened = [&neighbors, i] __device__(const Node& node)
					{
						const bool positionEqual = node.pos[0] == neighbors[i].pos[0] &&
							node.pos[1] == neighbors[i].pos[1];

						if (!positionEqual)
						{
							return false;
						}

						const bool betterFunctionG = neighbors[i].distanceFromBeginning < node.distanceFromBeginning;

						return betterFunctionG;
					};
					
					timer.reset();
					const auto betterNode = openNodes.find_if(findOpened);
					timer.printTime("	Find better opened node + lambda %f ms\n");

					if (betterNode != nullptr)
					{
						timer.reset();
						openNodes.remove(betterNode);
						openNodes.insert(neighbors[i]);
						timer.printTime("	Node insertion after node deletion %f ms\n");
					}
					else
					{	
						timer.reset();
						openNodes.insert(neighbors[i]);
						timer.printTime("	Node insertion %f ms\n");
					}
				}
				loopTimer.printTime("While loop iteration %f ms\n");
			}

			if (!finishNode)
			{
				return;
			}

			const int pathLength = finishNode->distanceFromBeginning + 1;
			timer2.printTime("Kernel overall time %f ms\n");
			//printf("path length %d\nopened nodes %d\nclosed nodes %d\n", pathLength, openNodes.size, closedNodes.size);
			
			//finishNode->getPathLength(pathLength);
			//int* tab = new int[2 * pathLength]; //x,y * length
			//finishNode->recreatePath(tab, pathLength);
			//atomicAdd(memSize, pathLength * 2);

			//agents[0].pathOutput = tab;
			//agents[0].pathSize = pathLength;
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
				memcpy(pathBuffer + agents[i].indexBegin, agents[i].pathOutput, agents[i].pathSize * 2);
				pathIndex += agents[i].pathSize * 2;
				delete[] agents[i].pathOutput;
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

