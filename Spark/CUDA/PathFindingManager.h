#ifndef PATH_FINDING_MANAGER_CUH
#define PATH_FINDING_MANAGER_CUH

#include <deque>
#include <list>
#include <memory>
#include <map>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "Map.cuh"
#include "Enums.h"

namespace spark {
	class ActorAI;
	class NodeAI;

	class PathFindingManager
	{
	public:
		cuda::Map map;

		PathFindingManager(const PathFindingManager& p) = delete;
		PathFindingManager(const PathFindingManager&& p) = delete;
		PathFindingManager& operator=(const PathFindingManager& p) = delete;
		PathFindingManager&& operator=(const PathFindingManager&& p) = delete;

		static PathFindingManager* getInstance();

		__host__ void addAgent(const std::shared_ptr<ActorAI>& agent);
		__host__ void findPaths();

	private:
		std::vector<std::weak_ptr<ActorAI>> agents;
		PathFindingMode mode = PathFindingMode::HOST;

		__host__ PathFindingManager();
		__host__ ~PathFindingManager() = default;

		__host__ void loadMap();
		__host__ void initializeMapOnGPU() const;
		__host__ void findPathsCUDA() const;
		__host__ std::uint16_t calculateNumberOfBlocks() const;
		__host__ std::uint8_t calculateNumberOfThreadsPerBlock(std::uint16_t numberOfBlocks) const;
		__host__ std::deque<glm::ivec2> findPath(const glm::ivec2 startPoint, const glm::ivec2 endPoint) const;
		__host__ NodeAI popFrom(std::multimap<float, NodeAI>& openedNodes) const;
		__host__ bool isNodeClosed(const std::list<NodeAI>& closedNodes, const NodeAI& node) const;
		__host__ void insertOrSwapNode(std::multimap<float, NodeAI>& openedNodes, float f, const NodeAI& node) const;
	};
}

#endif