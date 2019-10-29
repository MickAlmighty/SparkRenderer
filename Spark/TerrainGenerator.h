#ifndef TERRAIN_GENERATOR_H
#define TERRAIN_GENERATOR_H

#include <deque>

#include "Component.h"
#include "Structs.h"

namespace spark {

struct TerrainNode
{
	unsigned int numberOfActorsPassingThrough = 0;
	glm::vec3 nodeData {0};
};


class TerrainGenerator : public Component
{
public:
	int terrainSize = 20;

	TerrainGenerator(std::string&& newName = "TerrainGenerator");
	~TerrainGenerator();

	Texture generateTerrain();
	void updateTerrain() const;
	void markNodeAsPartOfPath(int x, int y);
	void unMarkNodeAsPartOfPath(int x, int y);
	float getTerrainValue(const int x, const int y);

	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;

	inline bool areIndexesValid(const int x, const int y) const
	{
		const bool validX = x >= 0 && x < terrainSize;
		const bool validY = y >= 0 && y < terrainSize;
		return validX && validY;
	}

private:
	float perlinDivider = 1.0f;
	float perlinTimeStep = 1.0f;
	Texture generatedTerrain{};
	std::vector<TerrainNode> terrain;

	int getTerrainNodeIndex(const int x, const int y) const;
	
};

}

#endif