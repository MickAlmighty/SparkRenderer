#ifndef PATH_FINDING_MANAGER_CUH
#define PATH_FINDING_MANAGER_CUH

#include <deque>
#include <memory>
#include <set>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "Map.cuh"
#include "Enums.h"
#include <thread>

namespace spark {
	class NodeAI;
	class ActorAI;

	class PathFindingManager
	{
	public:
		cuda::Map map;
		std::string mapPath;

		PathFindingManager(const PathFindingManager& p) = delete;
		PathFindingManager(const PathFindingManager&& p) = delete;
		PathFindingManager& operator=(const PathFindingManager& p) = delete;
		PathFindingManager&& operator=(const PathFindingManager&& p) = delete;

		static PathFindingManager* getInstance();

		__host__ void addAgent(const std::shared_ptr<ActorAI>& agent);
		__host__ void setMode(PathFindingMode implementationMode);
		__host__ void findPaths();
		__host__ void loadMap(const std::string& mapPath);
		__host__ void drawGui();
	private:
		std::vector<std::weak_ptr<ActorAI>> agents;
		PathFindingMode mode = PathFindingMode::DEVICE_IMPL_V2;

		__host__ PathFindingManager() = default;
		__host__ ~PathFindingManager() = default;

		__host__ void initializeMapOnGPU() const;
		__host__ void findPathsCUDA() const;
		__host__ void findPathsCPU() const;
		__host__ std::uint16_t calculateNumberOfBlocks(std::uint16_t maxThreadsPerBlock) const;
		__host__ std::uint16_t calculateNumberOfThreadsPerBlock(std::uint16_t numberOfBlocks) const;
		__host__ std::deque<glm::ivec2> findPath(const glm::ivec2 startPoint, const glm::ivec2 endPoint) const;
		__host__ void insertOrSwapNode(std::set<NodeAI>& openedNodes, const NodeAI& node) const;
	};
}

#endif