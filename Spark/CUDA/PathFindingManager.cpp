#include "PathFindingManager.h"

#include <stb_image/stb_image.h>

#include "ActorAI.h"
#include "DeviceMemory.h"
#include "kernel.cuh"
#include "Map.cuh"
#include "NodeAI.h"

namespace spark {

	PathFindingManager* PathFindingManager::getInstance()
	{
		static auto pathFindingManager = new PathFindingManager();
		return pathFindingManager;
	}

	void PathFindingManager::addAgent(const std::shared_ptr<ActorAI>& agent)
	{
		agents.push_back(agent);
	}

	void PathFindingManager::findPaths()
	{
		const double startTime = glfwGetTime();

		for (const auto& agent : agents)
		{
			if (agent.expired())
				continue;

			const auto agentPtr = agent.lock();
			auto path = findPath(agentPtr->startPos, agentPtr->endPos);
			agentPtr->setPath(path);
		}

		if (!agents.empty())
		{
			printf("Agents: %d, path finding time: %fms\n", static_cast<int>(agents.size()),
				(glfwGetTime() - startTime) * 1000.0f);
			agents.clear();
		}
	}

	PathFindingManager::PathFindingManager()
	{
		loadMap();
		initializeMapOnGPU();
	}

	void PathFindingManager::loadMap()
	{
		int texWidth, texHeight, nrChannels;

		unsigned char* pixels = stbi_load("map.png", &texWidth, &texHeight, &nrChannels, 0);

		std::vector<glm::vec3> pix;
		pix.reserve(texWidth * texHeight);
		for (int i = 0; i < texWidth * texHeight * nrChannels; i += 3)
		{
			glm::vec3 pixel;
			pixel.x = *(pixels + i);
			pixel.y = *(pixels + i + 1);
			pixel.z = *(pixels + i + 2);
			if (pixel != glm::vec3(0))
				pix.push_back(glm::normalize(pixel));
			else
				pix.push_back(pixel);
		}

		std::vector<float> mapNodes;
		mapNodes.resize(texWidth * texHeight);
		for (int i = 0; i < texWidth * texHeight; ++i)
		{
			mapNodes[i] = pix[i].x;
		}

		map = cuda::Map(texWidth, texHeight, mapNodes.data());
	}

	void PathFindingManager::initializeMapOnGPU() const
	{
		using namespace cuda;
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024); // max 1GB of GPU dynamic memory
		gpuErrchk(cudaGetLastError());

		const auto nodes = DeviceMemory<float>::AllocateElements(map.width * map.height);
		gpuErrchk(cudaGetLastError());
		cudaMemcpy(nodes.ptr, map.nodes, sizeof(float) * map.width * map.height, cudaMemcpyHostToDevice);
		gpuErrchk(cudaGetLastError());

		initMap(nodes.ptr, map.width, map.height);
	}

	std::deque<glm::ivec2> PathFindingManager::findPath(const glm::ivec2 startPoint, const glm::ivec2 endPoint) const
	{
		std::deque<glm::ivec2> path;
		std::multimap<float, NodeAI> openedNodes;
		std::list<NodeAI> closedNodes;

		openedNodes.emplace(0.0f, NodeAI(startPoint, 0.0f));
		NodeAI* finishNode = nullptr;
		while (true)
		{
			if (openedNodes.empty())
			{
				printf("Path hasn't been found\n");
				break;
			}

			const auto closedNode = popFrom(openedNodes);
			closedNodes.push_back(closedNode);

			if (closedNode.position == endPoint)
			{
				finishNode = &(*std::prev(closedNodes.end()));
				break;
			}

			auto neighbors = closedNode.getNeighbors(map);
			for (NodeAI& neighbor : neighbors)
			{
				if (isNodeClosed(closedNodes, neighbor))
					continue;

				neighbor.parentAddress = &(*std::prev(closedNodes.end()));

				const float functionH = neighbor.measureManhattanDistance(endPoint);
				const float functionG = neighbor.depth;

				const float terrainValue = map.getTerrainValue(neighbor.position.x, neighbor.position.y);
				const float heuristicsValue = (1 - terrainValue) * (functionH + functionG);

				insertOrSwapNode(openedNodes, heuristicsValue, neighbor);
			}
		}
		if (finishNode)
		{
			finishNode->getPath(path);
		}

		openedNodes.clear();
		closedNodes.clear();
		return path;
	}

	NodeAI PathFindingManager::popFrom(std::multimap<float, NodeAI>& openedNodes) const
	{
		const auto node_it = std::begin(openedNodes);
		if (node_it != std::end(openedNodes))
		{
			NodeAI n = node_it->second;
			openedNodes.erase(node_it);
			return n;
		}
		return {};
	}

	bool PathFindingManager::isNodeClosed(const std::list<NodeAI>& closedNodes, const NodeAI& node) const
	{
		const auto& it = std::find_if(std::begin(closedNodes), std::end(closedNodes), [&node](const NodeAI& n)
		{
			return n.position == node.position;
		});
		return it != std::end(closedNodes);
	}

	void PathFindingManager::insertOrSwapNode(std::multimap<float, NodeAI>& openedNodes, float f,
		const NodeAI& node) const
	{
		const auto& it = std::find_if(std::begin(openedNodes), std::end(openedNodes),
			[&node](const std::pair<float, NodeAI>& n)
		{
			const bool nodesEqual = n.second.position == node.position;
			if (!nodesEqual)
			{
				return false;
			}
			const bool betterFunctionG = node.depth < n.second.depth;

			return nodesEqual && betterFunctionG;
		});

		if (it != std::end(openedNodes))
		{
			openedNodes.erase(it);
			openedNodes.emplace(f, node);
		}
		else
		{
			openedNodes.emplace(f, node);
		}
	}
}
