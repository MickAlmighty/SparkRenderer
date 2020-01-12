#ifndef NODE_AI_H
#define NODE_AI_H

#include <deque>
#include <list>
#include <memory>

#include <glm/vec2.hpp>

#include "AgentSpawner.h"

namespace spark {
	namespace cuda {
		class Map;
	}

	class NodeAI
	{
	public:
		glm::ivec2 position;
		float depth = 0.0f;
		float functionF{ 0 };
		std::int32_t parentIdx{ -1 };

		NodeAI(const glm::ivec2 pos, const float depth_);
		NodeAI();
		~NodeAI() = default;

		bool operator<(const NodeAI& node) const
		{
			if (functionF == node.functionF)
			{
				return depth < node.depth;
			}
			else
				return functionF < node.functionF;
		}

		float measureDistanceTo(glm::ivec2 point) const;
		void calculateHeuristic(const cuda::Map& map, glm::ivec2 endPoint);
		std::vector<NodeAI> getNeighbors(const cuda::Map& map) const;

	private:
		inline void tryToCreateNeighbor(std::vector<NodeAI>& container, glm::ivec2&& pos,
			const cuda::Map& map, const float depth) const;
	};
}

#endif