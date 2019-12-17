#include "PathFindingManager.h"

#include <stb_image/stb_image.h>

#include "ActorAI.h"
#include "DeviceMemory.h"
#include "kernel.cuh"
#include "Map.cuh"
#include "Node.cuh"
#include "NodeAI.h"
#include <stb_image.h>

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

	void PathFindingManager::setMode(PathFindingMode implementationMode)
	{
		mode = implementationMode;
	}

	void PathFindingManager::findPaths()
	{
		PROFILE_FUNCTION();
		if (agents.empty())
			return;

		const double startTime = glfwGetTime();

		if (mode == PathFindingMode::HOST_IMPL)
		{
			findPathsCPU();
		}

		if (mode == PathFindingMode::DEVICE_IMPL || mode == PathFindingMode::DEVICE_IMPL_V2)
		{
			//const auto future = std::async(std::launch::async, [this] { findPathsCPU(); });
			findPathsCUDA();
		}

		const auto nextFrameAgentsCapacity = agents.capacity();
		agents.clear();
		agents.reserve(nextFrameAgentsCapacity);
	}

	void PathFindingManager::loadMap(const std::string& mapPath)
	{
		int texWidth, texHeight, nrChannels;
		float* pixels = stbi_loadf(mapPath.c_str(), &texWidth, &texHeight, &nrChannels, 0);

		std::vector<glm::vec3> pix;
		pix.reserve(texWidth * texHeight);
		for (int i = 0; i < texWidth * texHeight * nrChannels; i += nrChannels)
		{
			glm::vec3 pixel;
			pixel.x = *(pixels + i);
			pixel.y = *(pixels + i + 1);
			pixel.z = *(pixels + i + 2);
			pix.push_back(pixel);
		}

		std::vector<float> mapNodes;
		mapNodes.resize(texWidth * texHeight);
		for (int i = 0; i < texWidth * texHeight; ++i)
		{
			mapNodes[i] = pix[i].r;
		}

		map = cuda::Map(texWidth, texHeight, mapNodes.data());

		stbi_image_free(pixels);
		initializeMapOnGPU();
	}

	void PathFindingManager::drawGui()
	{
		static bool propertiesWindowOpened = false;
		if (ImGui::BeginMenu("PathFindingManager"))
		{
			std::string menuName = "Properties";
			if (ImGui::MenuItem(menuName.c_str()))
			{
				propertiesWindowOpened = true;
			}
			ImGui::EndMenu();
		}

		if (propertiesWindowOpened)
		{
			if (ImGui::Begin("Properties", &propertiesWindowOpened, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize))
			{
				static int implMode = static_cast<int>(mode);
				const auto r1 = ImGui::RadioButton("Host_impl", &implMode, static_cast<int>(PathFindingMode::HOST_IMPL)); ImGui::SameLine();
				const auto r2 = ImGui::RadioButton("Device_impl", &implMode, static_cast<int>(PathFindingMode::DEVICE_IMPL)); ImGui::SameLine();
				const auto r3 = ImGui::RadioButton("Device_impl_V2", &implMode, static_cast<int>(PathFindingMode::DEVICE_IMPL_V2));

				if (r1 || r2 || r3)
				{
					mode = static_cast<PathFindingMode>(implMode);
				}

				static const char* items[3] = { "map.png", "map2.png", "map3.png" };
				static int current_item = 0;
				if (ImGui::Combo("Map", &current_item, items, IM_ARRAYSIZE(items)))
				{
					loadMap(items[current_item]);
				}
				ImGui::End();
			}
		}
	}

	void PathFindingManager::initializeMapOnGPU() const
	{
		using namespace cuda;
		const auto nodes = DeviceMemory<float>::AllocateElements(map.width * map.height);
		gpuErrchk(cudaGetLastError());
		cudaMemcpy(nodes.ptr, map.nodes, sizeof(float) * map.width * map.height, cudaMemcpyHostToDevice);
		gpuErrchk(cudaGetLastError());

		initMap(nodes.ptr, map.width, map.height);
	}

	void PathFindingManager::findPathsCUDA() const
	{
		PROFILE_FUNCTION();
		std::uint16_t maxThreadsPerBlock = 32;
		auto threadMemoryMultiplier = 1;
		if (mode == PathFindingMode::DEVICE_IMPL)
		{ 
			threadMemoryMultiplier = 16;
		}
		if (mode == PathFindingMode::DEVICE_IMPL_V2)
		{
			threadMemoryMultiplier = 2;
			maxThreadsPerBlock = 256;
		}
			
		const auto blockCount = calculateNumberOfBlocks(maxThreadsPerBlock);
		const auto threadsPerBlock = calculateNumberOfThreadsPerBlock(blockCount);

		if (agents.size() < maxThreadsPerBlock)
			maxThreadsPerBlock = threadsPerBlock;

		struct path
		{
			glm::ivec2 start;
			glm::ivec2 end;
		};

		struct agentPathAssociation
		{
			uint32_t agentIdInAgentsVector;
			uint32_t pathIdInPathsVector;
		};
		std::vector<agentPathAssociation> associations;
		associations.reserve(agents.size());

		std::vector<path> pathsStartsEnds(maxThreadsPerBlock * blockCount);

		unsigned int actorId = 0;
		for (std::uint16_t blockId = 0; blockId < blockCount; ++blockId)
		{
			for (std::uint8_t threadId = 0; threadId < threadsPerBlock; ++threadId)
			{
				if (actorId == agents.size())
					break;
				pathsStartsEnds[blockId * maxThreadsPerBlock + threadId].start = agents[actorId].lock()->startPos;
				pathsStartsEnds[blockId * maxThreadsPerBlock + threadId].end = agents[actorId].lock()->endPos;
				associations.push_back({ static_cast<uint32_t>(actorId), static_cast<uint32_t>(blockId * maxThreadsPerBlock + threadId) });
				++actorId;
			}
		}

		Timer t("findPathsCUDA() GPU mem alloc and copy");
		const auto pathsStartsEndsSizeInBytes = pathsStartsEnds.size() * sizeof(path);
		const auto agentPathsSizeInBytes = map.width * map.height * sizeof(glm::ivec2) * blockCount * maxThreadsPerBlock;
		
		const int nodeSize = sizeof(cuda::Node);
		const auto kernelMemoryInBytes = nodeSize * map.width * map.height * maxThreadsPerBlock * blockCount * threadMemoryMultiplier;
		const auto overallBufferSizeInBytes = pathsStartsEndsSizeInBytes + agentPathsSizeInBytes + kernelMemoryInBytes;
		const auto devMem = cuda::DeviceMemory<std::uint8_t>::AllocateBytes(overallBufferSizeInBytes);
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(devMem.ptr, pathsStartsEnds.data(), pathsStartsEndsSizeInBytes, cudaMemcpyHostToDevice);
		gpuErrchk(cudaGetLastError());
		cudaDeviceSynchronize();
		t.stop();

		//printf("GPU allocation of %f mb memory\n", overallBufferSizeInBytes / 1024.0f / 1024.0f);

		const auto pathsStartsEndsDev = reinterpret_cast<int*>(devMem.ptr + 0);
		const auto agentPathsDev = reinterpret_cast<unsigned int*>(devMem.ptr + pathsStartsEndsSizeInBytes);
		const auto kernelMemoryDev = reinterpret_cast<void*>(devMem.ptr + pathsStartsEndsSizeInBytes + agentPathsSizeInBytes);
		
		{
			PROFILE_SCOPE("CUDA PathFinding Kernel");
			cuda::runKernel(blockCount, threadsPerBlock, pathsStartsEndsDev, agentPathsDev, kernelMemoryDev, map, mode, maxThreadsPerBlock);
			cudaDeviceSynchronize();
		}
		const auto agentPathSize = map.width * map.height * 2;
		const auto agentPathsBufferSize = agentPathSize * blockCount * maxThreadsPerBlock;
		std::vector<unsigned int> paths(agentPathsBufferSize);
		{
			PROFILE_SCOPE("Copying paths from GPU");
			cudaMemcpy(paths.data(), agentPathsDev, agentPathsBufferSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		}
		std::vector<std::vector<glm::ivec2>> pathsForAgents(blockCount * maxThreadsPerBlock);
		for (int i = 0; i < blockCount * maxThreadsPerBlock; ++i)
		{
			const auto pathSize = paths[agentPathSize * i];
			if (pathSize != 0)
			{
				pathsForAgents[i].resize(pathSize);
				memcpy(pathsForAgents[i].data(), reinterpret_cast<int*>(paths.data()) + agentPathSize * i + 1, sizeof(int) *  pathSize * 2);
			}
		}

		for (const auto& association : associations)
		{
			if (pathsForAgents[association.pathIdInPathsVector].empty())
			{
				std::cout << "There is no path for agent!" << std::endl;
			}
			else
			{
				agents[association.agentIdInAgentsVector].lock()->setPath(pathsForAgents[association.pathIdInPathsVector]);
			}
		}
	}

	void PathFindingManager::findPathsCPU() const
	{
		PROFILE_FUNCTION();
		for (const auto& agent : agents)
		{
			if (agent.expired())
				continue;

			const auto agentPtr = agent.lock();
			auto path = findPath(agentPtr->startPos, agentPtr->endPos);
			agentPtr->setPath(path);
		}
	}

	std::uint16_t PathFindingManager::calculateNumberOfBlocks(std::uint16_t maxThreadsPerBlock) const
	{
		std::uint16_t blockCount;
		if (agents.size() % maxThreadsPerBlock != 0)
		{
			blockCount = static_cast<std::uint16_t>(agents.size() / maxThreadsPerBlock + 1);
		}
		else
		{
			blockCount = static_cast<std::uint16_t>(agents.size() / maxThreadsPerBlock);
		}
		return blockCount;
	}

	std::uint16_t PathFindingManager::calculateNumberOfThreadsPerBlock(std::uint16_t numberOfBlocks) const
	{
		std::uint16_t threadsPerBlock;
		if (numberOfBlocks == 1)
		{
			threadsPerBlock = static_cast<std::uint16_t>(agents.size());
		}
		else
		{
			if (agents.size() % 2 != 0)
			{
				threadsPerBlock = static_cast<std::uint16_t>((agents.size() + 1) / numberOfBlocks);
			}
			else
			{
				threadsPerBlock = static_cast<std::uint16_t>(agents.size() / numberOfBlocks);
			}
		}

		return threadsPerBlock;
	}

	std::deque<glm::ivec2> PathFindingManager::findPath(const glm::ivec2 startPoint, const glm::ivec2 endPoint) const
	{
		std::deque<glm::ivec2> path;
		std::set<NodeAI> openedNodes;
		std::list<NodeAI> closedNodes;

		std::vector<std::vector<bool>> closedNodesTable(map.width);
		for (auto& cols : closedNodesTable)
		{
			cols.resize(map.height);
		}

		openedNodes.emplace(NodeAI(startPoint, 0.0f));
		NodeAI* finishNode = nullptr;

		while (!finishNode)
		{
			if (openedNodes.empty())
			{
				printf("Path hasn't been found\n");
				break;
			}

			//const auto closedNode = popFrom(openedNodes);
			const auto closedNode = *openedNodes.begin();
			openedNodes.erase(openedNodes.begin());

			closedNodes.push_back(closedNode);

			closedNodesTable[closedNode.position.x][closedNode.position.y] = true;

			auto neighbors = closedNode.getNeighbors(map);
			for (NodeAI& neighbor : neighbors)
			{
				/*if (isNodeClosed(closedNodes, neighbor))
					continue;*/
				if (closedNodesTable[neighbor.position.x][neighbor.position.y])
					continue;

				neighbor.parentAddress = &(*std::prev(closedNodes.end()));

				if (neighbor.position == endPoint)
				{
					closedNodes.push_back(neighbor);
					finishNode = &(*std::prev(closedNodes.end()));
					break;
				}

				neighbor.calculateHeuristic(map, endPoint);
				insertOrSwapNode(openedNodes, neighbor);
			}
		}
		if (finishNode)
		{
			finishNode->getPath(path);
		}

		//printf("CPU: Nodes processed %d, nodesToProcess %d, pathSize %d\n", 
			//static_cast<int>(closedNodes.size()) - 1, static_cast<int>(openedNodes.size()), static_cast<int>(path.size()));
		openedNodes.clear();
		closedNodes.clear();
		return path;
	}

	void PathFindingManager::insertOrSwapNode(std::set<NodeAI>& openedNodes, const NodeAI& node) const
	{
		const auto& it = std::find_if(std::begin(openedNodes), std::end(openedNodes),
			[&node](const NodeAI& n)
		{
			const bool nodesEqual = n.position == node.position;
			if (!nodesEqual)
			{
				return false;
			}
			const bool betterFunctionG = node.depth < n.depth;

			return nodesEqual && betterFunctionG;
		});

		if (it != std::end(openedNodes))
		{
			openedNodes.erase(it);
			openedNodes.insert(node);
		}

		openedNodes.insert(node);
	}
}
