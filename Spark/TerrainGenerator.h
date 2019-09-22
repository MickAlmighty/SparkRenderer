#pragma once
#include <Component.h>
#include "Structs.h"

class TerrainGenerator : public Component
{
private:
	int terrainSize = 20;
	float perlinDivider = 1.0f;
	float perlinTimeStep = 1.0f;
	std::vector<float> perlinValues;
public:
	SerializableType getSerializableType() override;
	Json::Value serialize() override;
	void deserialize(Json::Value& root) override;
	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;
	TerrainGenerator(std::string&& newName = "TerrainGenerator");
	~TerrainGenerator();

	Texture generateTerrain();
};

