#include "CUDA/kernel.cuh"

#include <device_launch_parameters.h>
#include <nvfunctional>
//#include <thrust/unique.h>
//#include <thrust/remove.h>
//#include <thrust/find.h>
//#include <thrust/execution_policy.h>

#include "BinaryHeap.cuh"
#include "DeviceMemory.h"
#include "DeviceTimer.cuh"
#include "Map.cuh"
#include "MemoryAllocator.cuh"
#include "MemoryManager.cuh"
#include "Node.cuh"
#include "Enums.h"
#include "Timer.h"


namespace spark {
	namespace cuda {
		__device__ Map* map = nullptr;

		__host__ void runKernel(int blocks, int threads, int* path, unsigned int* agentPaths, void* kernelMemory, const Map& mapHost, PathFindingMode mode, const int maxThreadsPerBlock)
		{
			int sharedMemorySize = maxThreadsPerBlock * sizeof(Node);
			if (mode == PathFindingMode::DEVICE_IMPL)
			{
				findPath<<<blocks, threads, sharedMemorySize>>>(path, agentPaths, kernelMemory, maxThreadsPerBlock);
			}
			if (mode == PathFindingMode::IMPL_BOTH)
			{
				findPath<<<blocks, threads, sharedMemorySize >> > (path, agentPaths, kernelMemory, maxThreadsPerBlock, true);
			}
			gpuErrchk(cudaGetLastError());
		}

		__host__ void initMap(float* nodes, int width, int height)
		{
			PROFILE_FUNCTION();
			createMap << <1, 1 >> > (nodes, width, height);
			gpuErrchk(cudaGetLastError());
		}

		__device__ void constructPath(unsigned int* path, Node* const finishNode, Node* closedNodes, bool showPathInfo)
		{
			int pathLength = 0;
			Node* node = finishNode;
			while (node->parentIdx != 65'535)
			{
				++pathLength;
				node = &closedNodes[node->parentIdx];
			}
			path[0] = pathLength;

			node = finishNode;
			int index = pathLength;
			while (node->parentIdx != 65'535)
			{
				index -= 1;
				path[1 + index * 2] = node->pos[0];
				path[1 + index * 2 + 1] = node->pos[1];
				node = &closedNodes[node->parentIdx];
			}
			if (showPathInfo)
				printf(" pathSize %d\n", pathLength + 1);
		}

		__global__  void createMap(float* nodes, int width, int height)
		{
			if (map != nullptr)
			{
				delete map;
			}
			map = new Map(width, height, nodes);
		}

		__global__ void findPath(int* path, unsigned* agentPaths, void* kernelMemory, const int maxThreadsPerBlock, bool showPathInfo)
		{
			extern __shared__ unsigned char sharedMemory[];

			const int mapSize = map->width * map->height;

			const unsigned int startEndPointsBlockMemorySize = 4 * maxThreadsPerBlock;
			const unsigned int startEndPointsThreadOffset = startEndPointsBlockMemorySize * blockIdx.x + 4 * threadIdx.
				x;
			int startPoint[] = { path[startEndPointsThreadOffset + 0], path[startEndPointsThreadOffset + 1] };
			int endPoint[] = { path[startEndPointsThreadOffset + 2], path[startEndPointsThreadOffset + 3] };

			if (startPoint[0] == endPoint[0] && startPoint[1] == endPoint[1])
			{
				//printf("start == end");
				return;
			}

			Node* neighbors = reinterpret_cast<Node*>(sharedMemory) + threadIdx.x; // 8 is the neighbor array size

			const auto agentPathMemorySize = mapSize * 2;
			const auto agentPathWarpMemorySize = agentPathMemorySize * maxThreadsPerBlock;
			const int agentPathOffset = agentPathWarpMemorySize * blockIdx.x + agentPathMemorySize * threadIdx.x;
			unsigned int* agentPath = agentPaths + agentPathOffset;

			const auto memoryOffsetInBytes = (sizeof(cuda::Node) + sizeof(uint32_t))* mapSize;
			const auto threadMemOffsetInBytes = memoryOffsetInBytes * threadIdx.x;
			const auto blockSizeElementsInBytes = memoryOffsetInBytes * maxThreadsPerBlock;
			const auto blockMemOffsetInBytes = blockSizeElementsInBytes * blockIdx.x;
			const auto threadMemoryBegin = static_cast<uint8_t*>(kernelMemory) + blockMemOffsetInBytes + threadMemOffsetInBytes;

			Node* closedNodes = reinterpret_cast<Node*>(threadMemoryBegin);
			MemoryManager manager = MemoryManager(closedNodes);

			int32_t* closedNodesIndices = reinterpret_cast<int32_t*>(closedNodes + mapSize);
			// it is more than needed now but it need to be equal to map width * map height
			memset(closedNodesIndices, -1, mapSize * sizeof(int32_t));
			BinaryHeap<uint16_t> heap(reinterpret_cast<uint16_t*>(agentPath),
				[&closedNodes] (const uint16_t& lhs, const uint16_t& rhs)
			{
				return closedNodes[lhs] < closedNodes[rhs];
			});

			Node* finishNode = nullptr;

			const Node startNode(startPoint, 0.0f);
			
			{
				const int startNodeIdx = map->getTerrainNodeIndex(startNode.pos[0], startNode.pos[1]);
				const auto closedStartNode = manager.allocate<Node>(startNode);
				const int closedNodeIdx = closedStartNode - closedNodes;
				closedNodesIndices[startNodeIdx] = closedNodeIdx;
				heap.insert(closedNodeIdx);
			}

			int whileLoopCounter = 0;
			while (!finishNode)
			{
				++whileLoopCounter;
				if (heap.size == 0)
				{
					printf("Heap is empty, nodes processed %d\n", whileLoopCounter);
					break;
				}

				if (heap.size >= map->width * map->height * 4)
				{
					printf("Heap size = %d limit %d reached!\n", heap.size, map->width * map->height);
					break;
				}

				auto theBestNode = closedNodes[heap.pop_front()];
				//theBestNode.getNeighbors(map, neighbors);

//#pragma unroll
//				for (int i = 0; i < 8; ++i)
//				{
//					if (neighbors[i].valid == 0)
//					{
//						continue;
//					}
//
//					const int nodeIdx = map->getTerrainNodeIndex(neighbors[i].pos[0], neighbors[i].pos[1]);
//					const int parentIndex = map->getTerrainNodeIndex(theBestNode.pos[0], theBestNode.pos[1]);
//					const int neighborIdxToClosedNodes = closedNodesIndices[nodeIdx];
//					const int parentIdxToClosedNodes = closedNodesIndices[parentIndex];
//					if (neighborIdxToClosedNodes != -1)
//					{
//						if (neighbors[i].distanceFromBeginning < closedNodes[neighborIdxToClosedNodes].distanceFromBeginning)
//						{
//							closedNodes[neighborIdxToClosedNodes].parentIdx = parentIdxToClosedNodes;
//							closedNodes[neighborIdxToClosedNodes].distanceFromBeginning = neighbors[i].distanceFromBeginning;
//
//							neighbors[i].calculateHeuristic(map, endPoint);
//							heap.insert(neighborIdxToClosedNodes);
//						}
//						continue;
//					}
//
//					neighbors[i].parentIdx = parentIdxToClosedNodes;
//					neighbors[i].calculateHeuristic(map, endPoint);
//					const auto closedNode = manager.allocate<Node>(neighbors[i]);
//
//					closedNodesIndices[nodeIdx] = closedNode - closedNodes; // info that node is closed
//					heap.insert(closedNode - closedNodes);
//
//					if (neighbors[i].pos[0] == endPoint[0] &&
//						neighbors[i].pos[1] == endPoint[1])
//					{
//						finishNode = closedNode;
//						break;
//					}
//				}


				for (int i = 0; i < 1; ++i)
				{
					theBestNode.tryToCreateNeighbor(neighbors, { theBestNode.pos[0] - 1, theBestNode.pos[1] }, map, 1);

					if (neighbors[0].valid == 0)
					{
						continue;
					}

					const int nodeIdx = map->getTerrainNodeIndex(neighbors[0].pos[0], neighbors[0].pos[1]);
					const int parentIndex = map->getTerrainNodeIndex(theBestNode.pos[0], theBestNode.pos[1]);
					const int neighborIdxToClosedNodes = closedNodesIndices[nodeIdx];
					const int parentIdxToClosedNodes = closedNodesIndices[parentIndex];
					if (neighborIdxToClosedNodes != -1)
					{
						if (neighbors[0].distanceFromBeginning < closedNodes[neighborIdxToClosedNodes].distanceFromBeginning)
						{
							closedNodes[neighborIdxToClosedNodes].parentIdx = parentIdxToClosedNodes;
							closedNodes[neighborIdxToClosedNodes].distanceFromBeginning = neighbors[0].distanceFromBeginning;

							neighbors[0].calculateHeuristic(map, endPoint);
							heap.insert(neighborIdxToClosedNodes);
						}
						continue;
					}

					neighbors[0].parentIdx = parentIdxToClosedNodes;
					neighbors[0].calculateHeuristic(map, endPoint);
					const auto closedNode = manager.allocate<Node>(neighbors[0]);

					closedNodesIndices[nodeIdx] = closedNode - closedNodes; // info that node is closed
					heap.insert(closedNode - closedNodes);

					if (neighbors[0].pos[0] == endPoint[0] &&
						neighbors[0].pos[1] == endPoint[1])
					{
						finishNode = closedNode;
						break;
					}
				}

				for (int i = 0; i < 1; ++i)
				{
					theBestNode.tryToCreateNeighbor(neighbors, { theBestNode.pos[0] + 1, theBestNode.pos[1] }, map, 1);

					if (neighbors[0].valid == 0)
					{
						continue;
					}

					const int nodeIdx = map->getTerrainNodeIndex(neighbors[0].pos[0], neighbors[0].pos[1]);
					const int parentIndex = map->getTerrainNodeIndex(theBestNode.pos[0], theBestNode.pos[1]);
					const int neighborIdxToClosedNodes = closedNodesIndices[nodeIdx];
					const int parentIdxToClosedNodes = closedNodesIndices[parentIndex];
					if (neighborIdxToClosedNodes != -1)
					{
						if (neighbors[0].distanceFromBeginning < closedNodes[neighborIdxToClosedNodes].distanceFromBeginning)
						{
							closedNodes[neighborIdxToClosedNodes].parentIdx = parentIdxToClosedNodes;
							closedNodes[neighborIdxToClosedNodes].distanceFromBeginning = neighbors[0].distanceFromBeginning;

							neighbors[0].calculateHeuristic(map, endPoint);
							heap.insert(neighborIdxToClosedNodes);
						}
						continue;
					}

					neighbors[0].parentIdx = parentIdxToClosedNodes;
					neighbors[0].calculateHeuristic(map, endPoint);
					const auto closedNode = manager.allocate<Node>(neighbors[0]);

					closedNodesIndices[nodeIdx] = closedNode - closedNodes; // info that node is closed
					heap.insert(closedNode - closedNodes);

					if (neighbors[0].pos[0] == endPoint[0] &&
						neighbors[0].pos[1] == endPoint[1])
					{
						finishNode = closedNode;
						break;
					}
				}

				for (int i = 0; i < 1; ++i)
				{
					theBestNode.tryToCreateNeighbor(neighbors, { theBestNode.pos[0], theBestNode.pos[1] - 1 }, map, 1);

					if (neighbors[0].valid == 0)
					{
						continue;
					}

					const int nodeIdx = map->getTerrainNodeIndex(neighbors[0].pos[0], neighbors[0].pos[1]);
					const int parentIndex = map->getTerrainNodeIndex(theBestNode.pos[0], theBestNode.pos[1]);
					const int neighborIdxToClosedNodes = closedNodesIndices[nodeIdx];
					const int parentIdxToClosedNodes = closedNodesIndices[parentIndex];
					if (neighborIdxToClosedNodes != -1)
					{
						if (neighbors[0].distanceFromBeginning < closedNodes[neighborIdxToClosedNodes].distanceFromBeginning)
						{
							closedNodes[neighborIdxToClosedNodes].parentIdx = parentIdxToClosedNodes;
							closedNodes[neighborIdxToClosedNodes].distanceFromBeginning = neighbors[0].distanceFromBeginning;

							neighbors[0].calculateHeuristic(map, endPoint);
							heap.insert(neighborIdxToClosedNodes);
						}
						continue;
					}

					neighbors[0].parentIdx = parentIdxToClosedNodes;
					neighbors[0].calculateHeuristic(map, endPoint);
					const auto closedNode = manager.allocate<Node>(neighbors[0]);

					closedNodesIndices[nodeIdx] = closedNode - closedNodes; // info that node is closed
					heap.insert(closedNode - closedNodes);

					if (neighbors[0].pos[0] == endPoint[0] &&
						neighbors[0].pos[1] == endPoint[1])
					{
						finishNode = closedNode;
						break;
					}
				}

				for (int i = 0; i < 1; ++i)
				{
					theBestNode.tryToCreateNeighbor(neighbors, { theBestNode.pos[0], theBestNode.pos[1] + 1 }, map, 1);

					if (neighbors[0].valid == 0)
					{
						continue;
					}

					const int nodeIdx = map->getTerrainNodeIndex(neighbors[0].pos[0], neighbors[0].pos[1]);
					const int parentIndex = map->getTerrainNodeIndex(theBestNode.pos[0], theBestNode.pos[1]);
					const int neighborIdxToClosedNodes = closedNodesIndices[nodeIdx];
					const int parentIdxToClosedNodes = closedNodesIndices[parentIndex];
					if (neighborIdxToClosedNodes != -1)
					{
						if (neighbors[0].distanceFromBeginning < closedNodes[neighborIdxToClosedNodes].distanceFromBeginning)
						{
							closedNodes[neighborIdxToClosedNodes].parentIdx = parentIdxToClosedNodes;
							closedNodes[neighborIdxToClosedNodes].distanceFromBeginning = neighbors[0].distanceFromBeginning;

							neighbors[0].calculateHeuristic(map, endPoint);
							heap.insert(neighborIdxToClosedNodes);
						}
						continue;
					}

					neighbors[0].parentIdx = parentIdxToClosedNodes;
					neighbors[0].calculateHeuristic(map, endPoint);
					const auto closedNode = manager.allocate<Node>(neighbors[0]);

					closedNodesIndices[nodeIdx] = closedNode - closedNodes; // info that node is closed
					heap.insert(closedNode - closedNodes);

					if (neighbors[0].pos[0] == endPoint[0] &&
						neighbors[0].pos[1] == endPoint[1])
					{
						finishNode = closedNode;
						break;
					}
				}

				for (int i = 0; i < 1; ++i)
				{
					theBestNode.tryToCreateNeighbor(neighbors, { theBestNode.pos[0] - 1, theBestNode.pos[1] - 1 }, map, 1.41f);

					if (neighbors[0].valid == 0)
					{
						continue;
					}

					const int nodeIdx = map->getTerrainNodeIndex(neighbors[0].pos[0], neighbors[0].pos[1]);
					const int parentIndex = map->getTerrainNodeIndex(theBestNode.pos[0], theBestNode.pos[1]);
					const int neighborIdxToClosedNodes = closedNodesIndices[nodeIdx];
					const int parentIdxToClosedNodes = closedNodesIndices[parentIndex];
					if (neighborIdxToClosedNodes != -1)
					{
						if (neighbors[0].distanceFromBeginning < closedNodes[neighborIdxToClosedNodes].distanceFromBeginning)
						{
							closedNodes[neighborIdxToClosedNodes].parentIdx = parentIdxToClosedNodes;
							closedNodes[neighborIdxToClosedNodes].distanceFromBeginning = neighbors[0].distanceFromBeginning;

							neighbors[0].calculateHeuristic(map, endPoint);
							heap.insert(neighborIdxToClosedNodes);
						}
						continue;
					}

					neighbors[0].parentIdx = parentIdxToClosedNodes;
					neighbors[0].calculateHeuristic(map, endPoint);
					const auto closedNode = manager.allocate<Node>(neighbors[0]);

					closedNodesIndices[nodeIdx] = closedNode - closedNodes; // info that node is closed
					heap.insert(closedNode - closedNodes);

					if (neighbors[0].pos[0] == endPoint[0] &&
						neighbors[0].pos[1] == endPoint[1])
					{
						finishNode = closedNode;
						break;
					}
				}

				for (int i = 0; i < 1; ++i)
				{
					theBestNode.tryToCreateNeighbor(neighbors, { theBestNode.pos[0] + 1, theBestNode.pos[1] - 1 }, map, 1.41f);

					if (neighbors[0].valid == 0)
					{
						continue;
					}

					const int nodeIdx = map->getTerrainNodeIndex(neighbors[0].pos[0], neighbors[0].pos[1]);
					const int parentIndex = map->getTerrainNodeIndex(theBestNode.pos[0], theBestNode.pos[1]);
					const int neighborIdxToClosedNodes = closedNodesIndices[nodeIdx];
					const int parentIdxToClosedNodes = closedNodesIndices[parentIndex];
					if (neighborIdxToClosedNodes != -1)
					{
						if (neighbors[0].distanceFromBeginning < closedNodes[neighborIdxToClosedNodes].distanceFromBeginning)
						{
							closedNodes[neighborIdxToClosedNodes].parentIdx = parentIdxToClosedNodes;
							closedNodes[neighborIdxToClosedNodes].distanceFromBeginning = neighbors[0].distanceFromBeginning;

							neighbors[0].calculateHeuristic(map, endPoint);
							heap.insert(neighborIdxToClosedNodes);
						}
						continue;
					}

					neighbors[0].parentIdx = parentIdxToClosedNodes;
					neighbors[0].calculateHeuristic(map, endPoint);
					const auto closedNode = manager.allocate<Node>(neighbors[0]);

					closedNodesIndices[nodeIdx] = closedNode - closedNodes; // info that node is closed
					heap.insert(closedNode - closedNodes);

					if (neighbors[0].pos[0] == endPoint[0] &&
						neighbors[0].pos[1] == endPoint[1])
					{
						finishNode = closedNode;
						break;
					}
				}

				for (int i = 0; i < 1; ++i)
				{
					theBestNode.tryToCreateNeighbor(neighbors, { theBestNode.pos[0] + 1, theBestNode.pos[1] + 1 }, map, 1.41f);

					if (neighbors[0].valid == 0)
					{
						continue;
					}

					const int nodeIdx = map->getTerrainNodeIndex(neighbors[0].pos[0], neighbors[0].pos[1]);
					const int parentIndex = map->getTerrainNodeIndex(theBestNode.pos[0], theBestNode.pos[1]);
					const int neighborIdxToClosedNodes = closedNodesIndices[nodeIdx];
					const int parentIdxToClosedNodes = closedNodesIndices[parentIndex];
					if (neighborIdxToClosedNodes != -1)
					{
						if (neighbors[0].distanceFromBeginning < closedNodes[neighborIdxToClosedNodes].distanceFromBeginning)
						{
							closedNodes[neighborIdxToClosedNodes].parentIdx = parentIdxToClosedNodes;
							closedNodes[neighborIdxToClosedNodes].distanceFromBeginning = neighbors[0].distanceFromBeginning;

							neighbors[0].calculateHeuristic(map, endPoint);
							heap.insert(neighborIdxToClosedNodes);
						}
						continue;
					}

					neighbors[0].parentIdx = parentIdxToClosedNodes;
					neighbors[0].calculateHeuristic(map, endPoint);
					const auto closedNode = manager.allocate<Node>(neighbors[0]);

					closedNodesIndices[nodeIdx] = closedNode - closedNodes; // info that node is closed
					heap.insert(closedNode - closedNodes);

					if (neighbors[0].pos[0] == endPoint[0] &&
						neighbors[0].pos[1] == endPoint[1])
					{
						finishNode = closedNode;
						break;
					}
				}

				for (int i = 0; i < 1; ++i)
				{
					theBestNode.tryToCreateNeighbor(neighbors, { theBestNode.pos[0] - 1, theBestNode.pos[1] + 1 }, map, 1.41f);

					if (neighbors[0].valid == 0)
					{
						continue;
					}

					const int nodeIdx = map->getTerrainNodeIndex(neighbors[0].pos[0], neighbors[0].pos[1]);
					const int parentIndex = map->getTerrainNodeIndex(theBestNode.pos[0], theBestNode.pos[1]);
					const int neighborIdxToClosedNodes = closedNodesIndices[nodeIdx];
					const int parentIdxToClosedNodes = closedNodesIndices[parentIndex];
					if (neighborIdxToClosedNodes != -1)
					{
						if (neighbors[0].distanceFromBeginning < closedNodes[neighborIdxToClosedNodes].distanceFromBeginning)
						{
							closedNodes[neighborIdxToClosedNodes].parentIdx = parentIdxToClosedNodes;
							closedNodes[neighborIdxToClosedNodes].distanceFromBeginning = neighbors[0].distanceFromBeginning;

							neighbors[0].calculateHeuristic(map, endPoint);
							heap.insert(neighborIdxToClosedNodes);
						}
						continue;
					}

					neighbors[0].parentIdx = parentIdxToClosedNodes;
					neighbors[0].calculateHeuristic(map, endPoint);
					const auto closedNode = manager.allocate<Node>(neighbors[0]);

					closedNodesIndices[nodeIdx] = closedNode - closedNodes; // info that node is closed
					heap.insert(closedNode - closedNodes);

					if (neighbors[0].pos[0] == endPoint[0] &&
						neighbors[0].pos[1] == endPoint[1])
					{
						finishNode = closedNode;
						break;
					}
				}
			}

			if (!finishNode)
			{
				return;
			}

			if(showPathInfo)
			{
				printf("GPU: Nodes processed %d, nodesToProcess %d,", whileLoopCounter, int(heap.size));
			}
			constructPath(agentPath, finishNode, closedNodes, showPathInfo);
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

