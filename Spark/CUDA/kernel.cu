#include "CUDA/kernel.cuh"

#include <deque>

#include <device_launch_parameters.h>
//#include <thrust/sort.h>
//#include <thrust/execution_policy.h>

#include "Node.cuh"
#include "List.cuh"
#include "Map.cuh"
#include "Agent.cuh"
#include "Timer.h"

namespace spark {
	namespace cuda {
		__device__ Map* map = nullptr;

		__host__ void runKernel(int* path, int* memSize, Agent* agents)
		{

			//checkMapValues <<<1, 1 >>> (map);
			cudaDeviceSynchronize();
			Timer t3("	findPath");
			findPath << <32, 32, 400 * sizeof(unsigned int) >> > (path, memSize, agents);
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

		__global__  void createMap(float* nodes, int width, int height)
		{
			if(map != nullptr)
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

			/*int integers[] = { 8, 5, 0, 2};
			thrust::sort(thrust::seq, integers, integers + 4);*/


			/*int startPoint[] = { *(path + 4 * threadIdx.x + 0), *(path + 4 * threadIdx.x + 1) };
			int endPoint[] = { *(path + 4 * threadIdx.x + 2), *(path + 4* threadIdx.x + 3) };*/

			int startPoint[] = { *(path + 0), *(path + 1) };
			int endPoint[] = { *(path + 2), *(path + 3) };

			const Node startNode(startPoint, 0.0f);
			List<Node> openNodes;
			List<Node> closedNodes;
			//Node* finishNode = nullptr;
			//openNodes.insert(startNode);

			//while(true)
			//{
			//	if(openNodes.size == 0)
			//	{
			//		break;
			//	}

			//	const auto closedNode = openNodes.pop_front();
			//	closedNodes.insert(closedNode);
			//	
			//	unsigned int beforeChange = closedNodesLookup[map->getTerrainNodeIndex(closedNode->value.pos[0], closedNode->value.pos[1])];
			//	const unsigned int change = beforeChange | (1 << threadIdx.x);
			//	closedNodesLookup[map->getTerrainNodeIndex(closedNode->value.pos[0], closedNode->value.pos[1])] = beforeChange | change;

			//	if(closedNode->value.pos[0] == endPoint[0] &&
			//		closedNode->value.pos[1] == endPoint[1])
			//	{
			//		finishNode = &closedNode->value;
			//		break;
			//	}

			//	const auto neighbors = closedNode->value.getNeighbors(map);

			//	for (auto neighborIt = neighbors.first; neighborIt != nullptr; neighborIt = neighborIt->next)
			//	{
			//		/*nvstd::function<bool(const Node& node)> f = [neighborIt] __device__ (const Node& node)
			//		{
			//			return node.pos[0] == neighborIt->value.pos[0] &&
			//				node.pos[1] == neighborIt->value.pos[1];
			//		};*/
			//		if (closedNodesLookup[map->getTerrainNodeIndex(neighborIt->value.pos[0], neighborIt->value.pos[1])] & (1 << threadIdx.x))
			//		{
			//			continue;
			//		}

			//		/*const auto neighborClosed = closedNodes.find(neighborIt->value);
			//		if (neighborClosed != nullptr)
			//			continue;*/

			//		neighborIt->value.parent = &closedNode->value;

			//		neighborIt->value.valueH = neighborIt->value.measureManhattanDistance(endPoint);
			//		const float functionG = neighborIt->value.distanceFromBeginning;
			//		const int neighborPos[2] = { neighborIt->value.pos[0], neighborIt->value.pos[1] };
			//		const float terrainValue = map->getTerrainValue(neighborPos[0], neighborPos[1]);
			//		neighborIt->value.valueF = (1.0f - terrainValue) * (neighborIt->value.valueH + functionG);
			//		

			//		nvstd::function<bool(const Node& node)> findOpened = [neighborIt] __device__ (const Node& node)
			//		{
			//			const bool positionEqual = node.pos[0] == neighborIt->value.pos[0] &&
			//				node.pos[1] == neighborIt->value.pos[1];
			//			
			//			if (!positionEqual)
			//			{
			//				return false;
			//			}

			//			const bool betterFunctionG = neighborIt->value.distanceFromBeginning < node.distanceFromBeginning;
			//			
			//			return betterFunctionG;
			//		};

			//		const auto betterNode = openNodes.find_if(findOpened);
			//		if (betterNode != nullptr)
			//		{
			//			openNodes.remove(betterNode);
			//			openNodes.insert(neighborIt->value);
			//		}
			//		else
			//		{
			//			openNodes.insert(neighborIt->value);
			//		}
			//	}
			//}
			//
			//if(!finishNode)
			//{
			//	return;
			//}

			//int pathLength = 0;
			//finishNode->getPathLength(pathLength);
			////int* tab = new int[2 * pathLength]; //x,y * length
			////finishNode->recreatePath(tab, pathLength);
			//atomicAdd(memSize, pathLength * 2);

			//agents[0].pathOutput = tab;
			//agents[0].pathSize = pathLength;
		}

		__global__ void fillPathBuffer(Agent* agents, int* pathBuffer, int numberOfAgents)
		{
			int pathIndex = 0;
			for(int i = 0; i < numberOfAgents; ++i)
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
	}
}

