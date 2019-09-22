#pragma once
#include <Component.h>
#include "Structs.h"

class TerrainGenerator : public Component
{
private:
	
	float perlinDivider = 1.0f;
	float perlinTimeStep = 1.0f;
	Texture generatedTerrain{};
public:
	std::vector<glm::vec3> perlinValues;
	int terrainSize = 20;

	SerializableType getSerializableType() override;
	Json::Value serialize() override;
	void deserialize(Json::Value& root) override;
	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;
	TerrainGenerator(std::string&& newName = "TerrainGenerator");
	~TerrainGenerator();

	Texture generateTerrain();
	void updateTerrain();
};

