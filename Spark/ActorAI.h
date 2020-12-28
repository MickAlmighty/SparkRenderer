#pragma once

#include "Component.h"
#include "NodeAI.h"

#include <glm/glm.hpp>

#include <deque>

namespace spark
{
class TerrainGenerator;

class ActorAI final : public Component
{
    public:
    ActorAI();
    ~ActorAI() = default;
    ActorAI(const ActorAI&) = delete;
    ActorAI(const ActorAI&&) = delete;
    ActorAI& operator=(const ActorAI&) = delete;
    ActorAI& operator=(const ActorAI&&) = delete;

    void update() override;
    void fixedUpdate() override;
    void drawGUI() override;

    private:
    double timer = 0.0f;
    float movementSpeed = 1.0f;
    bool isTraveling = false;
    glm::ivec2 startPos{};
    glm::ivec2 endPos{};
    std::multimap<float, std::shared_ptr<NodeAI>> nodesToProcess;
    std::multimap<float, NodeAI> nodesToProcessStack;

    std::list<std::shared_ptr<NodeAI>> processedNodes;
    std::list<NodeAI> processedNodesStack;

    std::deque<std::pair<bool, glm::ivec2>> path;
    std::weak_ptr<TerrainGenerator> terrainGenerator;

    void findPath();
    std::shared_ptr<NodeAI> getTheNearestNodeFromOpen();
    bool isNodeClosed(const std::shared_ptr<NodeAI>& node);
    void walkToEndOfThePath();

    void findPathStack();
    NodeAI getTheNearestNodeFromOpenStack();
    bool isNodeClosedStack(const NodeAI& node);

    void validateActorPosition(glm::vec3& position) const;
    void setStartPosition(glm::vec3& startPosition);
};
}  // namespace spark