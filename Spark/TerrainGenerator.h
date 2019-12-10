#ifndef TERRAIN_GENERATOR_H
#define TERRAIN_GENERATOR_H

#include "Component.h"
#include "Structs.h"

namespace spark {
	class TerrainGenerator : public Component
	{
	public:
		TerrainGenerator(std::string&& newName = "TerrainGenerator");
		~TerrainGenerator();

		Texture generateTerrain();
		void updateTerrain() const;

		SerializableType getSerializableType() override;
		Json::Value serialize() override;
		void deserialize(Json::Value& root) override;
		void update() override;
		void fixedUpdate() override;
		void drawGUI() override;

	private:
		int terrainSize = 20;
		std::size_t agentCounter{ 0 };

		Texture generatedTerrain{};
		std::vector<float> terrain;
	};

}

#endif