#include "CUDA/kernel.cuh"

#include <device_launch_parameters.h>
#include <nvfunctional>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>

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
		__device__ MemoryAllocator* allocator[128];

		__host__ void runKernel(int blocks, int threads, int* path, unsigned int* agentPaths, void* kernelMemory, const Map& mapHost, PathFindingMode mode, const int maxThreadsPerBlock)
		{
			if (mode == PathFindingMode::DEVICE_IMPL)
			{
				//findPathV1<<<blocks, threads, mapHost.width * mapHost.height * sizeof(unsigned int) + 32 * 8 * sizeof(Node)>>> (path, agentPaths, kernelMemory);
			}
			if (mode == PathFindingMode::DEVICE_IMPL_V2)
			{
				int sharedMemorySize = maxThreadsPerBlock * 8 * sizeof(Node);
				findPathV3<<<blocks, threads, sharedMemorySize>>> (path, agentPaths, kernelMemory, maxThreadsPerBlock);
			}
			gpuErrchk(cudaGetLastError());
		}

		__host__ void initMap(float* nodes, int width, int height)
		{
			PROFILE_FUNCTION();
			createMap<<<1, 1 >>>(nodes, width, height);
			gpuErrchk(cudaGetLastError());
		}

		__device__ void constructPath(unsigned int* path, Node* const finishNode, Node* closedNodes)
		{
			int pathLength = 0;
			Node* node = finishNode;
			while (node->parentIdx >= 0)
			{
				++pathLength;
				node = &closedNodes[node->parentIdx];
			}
			path[0] = pathLength;

			node = finishNode;
			int index = pathLength;
			while (node->parentIdx >= 0)
			{
				index -= 1;
				path[1 + index * 2] = node->pos[0];
				path[1 + index * 2 + 1] = node->pos[1];
				node = &closedNodes[node->parentIdx];
			}
		}

		__global__  void createMap(float* nodes, int width, int height)
		{
			if (map != nullptr)
			{
				delete map;
			}
			map = new Map(width, height, nodes);
		}

//		__global__ void findPathV1(int* path, unsigned int* agentPaths, void* kernelMemory)
//		{
//			extern __shared__ unsigned char sharedMemory[];
//
//			unsigned int* closedNodesLookup = reinterpret_cast<unsigned int*>(sharedMemory);
//
//			if (threadIdx.x == 0)
//			{
//				memset(sharedMemory, 0, sizeof(unsigned int) * map->width * map->height + 8 * 32 * sizeof(Node));
//			}
//			__syncthreads();
//
//			Node* neighborsMemoryPool = reinterpret_cast<Node*>(sharedMemory + map->width * map->height * sizeof(
//				unsigned int));
//			Node* neighbors = neighborsMemoryPool + threadIdx.x * 8; // 8 is the neighbor array size
//
//			const auto agentPathMemorySize = map->width * map->height * 2;
//			const auto agentPathWarpMemorySize = agentPathMemorySize * 32;
//			const int agentPathOffset = agentPathWarpMemorySize * blockIdx.x + agentPathMemorySize * threadIdx.x;
//			unsigned int* agentPath = agentPaths + agentPathOffset;
//
//			const auto memoryOffset = map->width * map->height * 16;
//			const auto threadMemOffset = memoryOffset * threadIdx.x;
//			const auto blockSizeElements = memoryOffset * 32;
//			const auto blockMemOffset = blockSizeElements * blockIdx.x;
//			BinaryHeap<Node> heap(static_cast<Node*>(kernelMemory) + blockMemOffset + threadMemOffset);
//			MemoryManager manager = MemoryManager(
//				static_cast<Node*>(kernelMemory) + blockMemOffset + threadMemOffset + map->width * map->height * 8);
//
//			const unsigned int startEndPointsBlockMemorySize = 4 * 32;
//			const unsigned int startEndPointsThreadOffset = startEndPointsBlockMemorySize * blockIdx.x + 4 * threadIdx.
//				x;
//			int startPoint[] = { path[startEndPointsThreadOffset + 0], path[startEndPointsThreadOffset + 1] };
//			int endPoint[] = { path[startEndPointsThreadOffset + 2], path[startEndPointsThreadOffset + 3] };
//
//			if (startPoint[0] == endPoint[0] && startPoint[1] == endPoint[1])
//				return;
//
//			Node* finishNode = nullptr;
//
//			const Node startNode(startPoint, 0.0f);
//			heap.insert(startNode);
//
//			int i = 0;
//			const nvstd::function<bool(const Node& node)> findOpened = [&neighbors, &i] __device__(const Node& node)
//			{
//				if (node.pos[0] == neighbors[i].pos[0] &&
//					node.pos[1] == neighbors[i].pos[1])
//				{
//					//betterFunctionG
//					return neighbors[i].distanceFromBeginning < node.distanceFromBeginning;
//				}
//				return false;
//			};
//
//			int whileLoopCounter = 0;
//			while (!finishNode)
//			{
//				++whileLoopCounter;
//				if (heap.size == 0)
//				{
//					printf("Heap is empty\n");
//					break;
//				}
//
//				if (heap.size >= map->width * map->height * 8)
//				{
//					printf("Heap size = %d limit %d reached!\n", heap.size, map->width * map->height * 8);
//					break;
//				}
//
//				if (whileLoopCounter >= map->width * map->height * 8)
//				{
//					printf("Closed nodes limit reached!\n");
//					break;
//				}
//
//				const auto end = thrust::unique(thrust::seq, heap.array, heap.array + heap.size);
//				heap.size = end - heap.array;
//				
//				auto theBestNode = heap.pop_front();
//
//				const auto closedNode = manager.allocate<Node>(theBestNode);
//
//				const int nodeIndex = map->getTerrainNodeIndex(closedNode->pos[0], closedNode->pos[1]);
//				atomicOr(closedNodesLookup + nodeIndex, 1 << threadIdx.x);
//
//				closedNode->getNeighbors(map, neighbors);
//
//#pragma unroll
//				for (i = 0; i < 8; ++i)
//				{
//					if (neighbors[i].valid == 0)
//					{
//						continue;
//					}
//
//					const int nodeIdx = map->getTerrainNodeIndex(neighbors[i].pos[0], neighbors[i].pos[1]);
//					if (closedNodesLookup[nodeIdx] & (1 << threadIdx.x))
//					{
//						continue;
//					}
//
//					neighbors[i].parent = closedNode;
//
//					if (neighbors[i].pos[0] == endPoint[0] &&
//						neighbors[i].pos[1] == endPoint[1])
//					{
//						finishNode = manager.allocate<Node>(neighbors[i]);
//						break;
//					}
//
//					const float functionG = neighbors[i].distanceFromBeginning;
//					const int neighborPos[2] = { neighbors[i].pos[0], neighbors[i].pos[1] };
//					const float terrainValue = map->getTerrainValue(neighborPos[0], neighborPos[1]);
//					neighbors[i].valueF = (1.0f - terrainValue) * (neighbors[i].measureDistanceTo(endPoint) + functionG
//					);
//
//					/*const auto lastNode = thrust::remove(thrust::device, heap.array, heap.array + heap.size, neighbors[i]);
//					heap.size = lastNode - heap.array;*/
//
//					const  auto nodeAddress = thrust::find(thrust::seq, heap.array, heap.array + heap.size,
//					                                       neighbors[i]);
//					if (nodeAddress - heap.array != heap.size)
//					{
//						heap.removeValue(nodeAddress - heap.array);
//					}
//					heap.insert(neighbors[i]);
//				}
//			}
//
//			if (!finishNode)
//			{
//				return;
//			}
//
//			constructPath(agentPath, finishNode);
//			//printf("GPU: Nodes processed %d, nodesToProcess %d, pathSize %d\n", whileLoopCounter, int(heap.size), pathLength);
//		}

		__global__ void findPathV3(int* path, unsigned* agentPaths, void* kernelMemory, const int maxThreadsPerBlock)
		{
			extern __shared__ unsigned char sharedMemory[];
			
			const int mapSize = map->width * map->height;

			const unsigned int startEndPointsBlockMemorySize = 4 * maxThreadsPerBlock;
			const unsigned int startEndPointsThreadOffset = startEndPointsBlockMemorySize * blockIdx.x + 4 * threadIdx.
				x;
			int startPoint[] = { path[startEndPointsThreadOffset + 0], path[startEndPointsThreadOffset + 1] };
			int endPoint[] = { path[startEndPointsThreadOffset + 2], path[startEndPointsThreadOffset + 3] };

			if (startPoint[0] == endPoint[0] && startPoint[1] == endPoint[1])
				return;

			Node* neighbors = reinterpret_cast<Node*>(sharedMemory) + threadIdx.x * 8; // 8 is the neighbor array size

			const auto agentPathMemorySize = mapSize * 2;
			const auto agentPathWarpMemorySize = agentPathMemorySize * maxThreadsPerBlock;
			const int agentPathOffset = agentPathWarpMemorySize * blockIdx.x + agentPathMemorySize * threadIdx.x;
			unsigned int* agentPath = agentPaths + agentPathOffset;

			const auto memoryOffset = mapSize * 3;
			const auto threadMemOffset = memoryOffset * threadIdx.x;
			const auto blockSizeElements = memoryOffset * maxThreadsPerBlock;
			const auto blockMemOffset = blockSizeElements * blockIdx.x;
			const auto threadMemoryBegin = static_cast<Node*>(kernelMemory) + blockMemOffset + threadMemOffset;
			BinaryHeap<Node> heap(threadMemoryBegin);
			Node* closedNodes = threadMemoryBegin + mapSize;
			MemoryManager manager = MemoryManager(closedNodes);

			int* closedNodesIndices = reinterpret_cast<int*>(threadMemoryBegin + mapSize * 2);
			// it is more than needed now but it need to be equal to map width * map height
			memset(closedNodesIndices, -1, mapSize * sizeof(int));

			Node* finishNode = nullptr;

			const Node startNode(startPoint, 0.0f);
			heap.insert(startNode);

			{
				const int startNodeIdx = map->getTerrainNodeIndex(startNode.pos[0], startNode.pos[1]);
				const auto closedStartNode = manager.allocate<Node>(startNode);
				const int closedNodeIdx = closedStartNode - closedNodes;
				closedNodesIndices[startNodeIdx] = closedNodeIdx;
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

				if (heap.size >= map->width * map->height)
				{
					printf("Heap size = %d limit %d reached!\n", heap.size, map->width * map->height);
					break;
				}

				auto theBestNode = heap.pop_front();
				theBestNode.getNeighbors(map, neighbors);
				
				#pragma unroll
				for (int i = 0; i < 8; ++i)
				{
					if (neighbors[i].valid == 0)
					{
						continue;
					}

					const int nodeIdx = map->getTerrainNodeIndex(neighbors[i].pos[0], neighbors[i].pos[1]);
					const int parentIndex = map->getTerrainNodeIndex(theBestNode.pos[0], theBestNode.pos[1]);
					if (closedNodesIndices[nodeIdx] != -1)
					{
						if (neighbors[i].distanceFromBeginning < closedNodes[closedNodesIndices[nodeIdx]].
							distanceFromBeginning)
						{
							closedNodes[closedNodesIndices[nodeIdx]].parentIdx = closedNodesIndices[parentIndex];
							closedNodes[closedNodesIndices[nodeIdx]].distanceFromBeginning = neighbors[i].distanceFromBeginning;
							
							neighbors[i].calculateHeuristic(map, endPoint);
							heap.insert(neighbors[i]);
						}
						continue;
					}

					neighbors[i].parentIdx = closedNodesIndices[parentIndex];
					neighbors[i].calculateHeuristic(map, endPoint);
					heap.insert(neighbors[i]);

					const auto closedNode = manager.allocate<Node>(neighbors[i]);
					closedNodesIndices[nodeIdx] = closedNode - closedNodes; // info that node is closed

					if (neighbors[i].pos[0] == endPoint[0] &&
						neighbors[i].pos[1] == endPoint[1])
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

			constructPath(agentPath, finishNode, closedNodes);
			//printf("GPU: Nodes processed %d, nodesToProcess %d, pathSize %d\n", whileLoopCounter, int(heap.size), pathLength);
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
//		__global__ void findPathV2(int* path, unsigned* agentPaths, void* kernelMemory)
//		{
//			extern __shared__ unsigned char sharedMemory[];
//
//			unsigned int* closedNodesLookup = reinterpret_cast<unsigned int*>(sharedMemory);
//
//			if (threadIdx.x == 0)
//			{
//				memset(sharedMemory, 0, sizeof(unsigned int) * map->width * map->height + 8 * 32 * sizeof(Node));
//			}
//			__syncthreads();
//
//			Node* neighborsMemoryPool = reinterpret_cast<Node*>(sharedMemory + map->width * map->height * sizeof(
//				unsigned int));
//			Node* neighbors = neighborsMemoryPool + threadIdx.x * 8; // 8 is the neighbor array size
//
//			const auto agentPathMemorySize = map->width * map->height * 2;
//			const auto agentPathWarpMemorySize = agentPathMemorySize * 32;
//			const int agentPathOffset = agentPathWarpMemorySize * blockIdx.x + agentPathMemorySize * threadIdx.x;
//			unsigned int* agentPath = agentPaths + agentPathOffset;
//
//			const auto memoryOffset = map->width * map->height * 2;
//			const auto threadMemOffset = memoryOffset * threadIdx.x;
//			const auto blockSizeElements = memoryOffset * 32;
//			const auto blockMemOffset = blockSizeElements * blockIdx.x;
//			BinaryHeap<Node> heap(static_cast<Node*>(kernelMemory) + blockMemOffset + threadMemOffset);
//			MemoryManager manager = MemoryManager(
//				static_cast<Node*>(kernelMemory) + blockMemOffset + threadMemOffset + map->width * map->height);
//
//			const unsigned int startEndPointsBlockMemorySize = 4 * 32;
//			const unsigned int startEndPointsThreadOffset = startEndPointsBlockMemorySize * blockIdx.x + 4 * threadIdx.
//				x;
//			int startPoint[] = { path[startEndPointsThreadOffset + 0], path[startEndPointsThreadOffset + 1] };
//			int endPoint[] = { path[startEndPointsThreadOffset + 2], path[startEndPointsThreadOffset + 3] };
//
//			if (startPoint[0] == endPoint[0] && startPoint[1] == endPoint[1])
//				return;
//
//			Node* finishNode = nullptr;
//
//			const Node startNode(startPoint, 0.0f);
//			heap.insert(startNode);
//
//			int i = 0;
//			const nvstd::function<bool(const Node& node)> findOpened = [&neighbors, &i] __device__(const Node& node)
//			{
//				if (node.pos[0] == neighbors[i].pos[0] &&
//					node.pos[1] == neighbors[i].pos[1])
//				{
//					//betterFunctionG
//					return neighbors[i].distanceFromBeginning < node.distanceFromBeginning;
//				}
//				return false;
//			};
//
//			int whileLoopCounter = 0;
//			while (!finishNode)
//			{
//				++whileLoopCounter;
//				if (heap.size == 0)
//				{
//					printf("Heap is empty\n");
//					break;
//				}
//
//				if (heap.size >= map->width * map->height)
//				{
//					printf("Heap size = %d limit %d reached!\n", heap.size, map->width * map->height * 8);
//					break;
//				}
//
//				if (whileLoopCounter >= map->width * map->height)
//				{
//					printf("Closed nodes limit reached!\n");
//					break;
//				}
//
//				auto theBestNode = heap.pop_front();
//
//				const auto closedNode = manager.allocate<Node>(theBestNode);
//
//				closedNode->getNeighbors(map, neighbors);
//#pragma unroll
//				for (i = 0; i < 8; ++i)
//				{
//					if (neighbors[i].valid == 0)
//					{
//						continue;
//					}
//
//					const int nodeIdx = map->getTerrainNodeIndex(neighbors[i].pos[0], neighbors[i].pos[1]);
//					if (closedNodesLookup[nodeIdx] & (1 << threadIdx.x))
//					{
//						continue;
//					}
//					atomicOr(closedNodesLookup + nodeIdx, 1 << threadIdx.x);
//
//					neighbors[i].parent = closedNode;
//
//					if (neighbors[i].pos[0] == endPoint[0] &&
//						neighbors[i].pos[1] == endPoint[1])
//					{
//						finishNode = manager.allocate<Node>(neighbors[i]);
//						break;
//					}
//
//					const float functionG = neighbors[i].distanceFromBeginning;
//					const int neighborPos[2] = { neighbors[i].pos[0], neighbors[i].pos[1] };
//					const float terrainValue = map->getTerrainValue(neighborPos[0], neighborPos[1]);
//					neighbors[i].valueF = (1.0f - terrainValue) * (neighbors[i].measureDistanceTo(endPoint) + functionG
//					);
//
//					heap.insert(neighbors[i]);
//				}
//			}
//
//			if (!finishNode)
//			{
//				return;
//			}
//
//			constructPath(agentPath, finishNode);
//			//printf("GPU: Nodes processed %d, nodesToProcess %d, pathSize %d\n", whileLoopCounter, int(heap.size), pathLength);
//		}
}

