#pragma once

#include <deque>
#include <list>
#include <memory>

#include <glm/vec2.hpp>

#include "TerrainGenerator.h"

namespace spark
{
class NodeAI final
{
    public:
    std::weak_ptr<NodeAI> parent;
    const glm::ivec2 position;
    float depth = 0.0f;
    NodeAI* parentAddress = nullptr;

    NodeAI(const glm::ivec2 pos, const float depth_);
    NodeAI(const NodeAI& rhs);
    NodeAI(const NodeAI&& rhs) noexcept;
    NodeAI();
    ~NodeAI() = default;

    float distanceToEndPoint(glm::vec2 endPoint) const;
    std::list<std::shared_ptr<NodeAI>> getNeighbors(const std::shared_ptr<TerrainGenerator>& terrainGenerator) const;
    void drawReturnPath(std::shared_ptr<TerrainGenerator>& terrainGenerator) const;
    void getPath(std::deque<std::pair<bool, glm::ivec2>>& path) const;

    std::list<NodeAI> getNeighborsStack(const std::shared_ptr<TerrainGenerator>& terrainGenerator) const;
    void drawReturnPathStack(std::shared_ptr<TerrainGenerator>& terrainGenerator) const;
    void getPathStack(std::deque<std::pair<bool, glm::ivec2>>& path) const;

    private:
    inline void tryToCreateNeighbor(std::list<std::shared_ptr<NodeAI>>& container, glm::ivec2&& pos,
                                    const std::shared_ptr<TerrainGenerator>& terrainGenerator, const float depth) const;
    inline void tryToCreateNeighbor(std::list<NodeAI>& container, glm::ivec2&& pos, const std::shared_ptr<TerrainGenerator>& terrainGenerator,
                                    const float depth) const;
};
}  // namespace spark
