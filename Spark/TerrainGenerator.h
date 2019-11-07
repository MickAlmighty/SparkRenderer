#ifndef TERRAIN_GENERATOR_H
#define TERRAIN_GENERATOR_H

#include <deque>

#include "Component.h"
#include "Structs.h"

namespace spark {
	class TerrainGenerator : public Component
	{
	public:
		int terrainSize = 20;

		TerrainGenerator(std::string&& newName = "TerrainGenerator");
		~TerrainGenerator();

		Texture generateTerrain();
		void updateTerrain() const;
		float getTerrainValue(const int x, const int y);

		SerializableType getSerializableType() override;
		Json::Value serialize() override;
		void deserialize(Json::Value& root) override;
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
		Texture generatedTerrain{};
		std::vector<float> terrain;
		int getTerrainNodeIndex(const int x, const int y) const;
	};

}

#endif