#pragma once

#include <deque>

#include "Component.h"
#include "Structs.h"

namespace spark
{
struct TerrainNode
{
    unsigned int numberOfActorsPassingThrough = 0;
    glm::vec3 nodeData{0};
};

class TerrainGenerator final : public Component
{
    public:
    TerrainGenerator();
    TerrainGenerator(std::string&& newName);
    ~TerrainGenerator();
    TerrainGenerator(const TerrainGenerator&) = delete;
    TerrainGenerator(TerrainGenerator&&) = delete;
    TerrainGenerator& operator=(const TerrainGenerator&) = delete;
    TerrainGenerator& operator=(TerrainGenerator&&) = delete;

    Texture generateTerrain();
    void updateTerrain() const;
    void markNodeAsPartOfPath(int x, int y);
    void unMarkNodeAsPartOfPath(int x, int y);
    float getTerrainValue(const int x, const int y);

    void update() override;
    void fixedUpdate() override;
    void drawGUI() override;

    bool areIndicesValid(const int x, const int y) const;

    int terrainSize{20};

    private:
    float perlinDivider = 1.0f;
    float perlinTimeStep = 1.0f;
    Texture generatedTerrain{};
    std::vector<TerrainNode> terrain;  // FIXME: should it be serialized too, or just regenerated?

    int getTerrainNodeIndex(const int x, const int y) const;
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(Component);
};

}  // namespace spark