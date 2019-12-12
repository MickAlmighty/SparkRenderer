#ifndef TERRAIN_GENERATOR_H
#define TERRAIN_GENERATOR_H

#include "Component.h"

namespace spark {
	class TerrainGenerator : public Component
	{
	public:
		TerrainGenerator(std::string&& newName = "TerrainGenerator");
		~TerrainGenerator() = default;

		SerializableType getSerializableType() override;
		Json::Value serialize() override;
		void deserialize(Json::Value& root) override;
		void update() override;
		void fixedUpdate() override;
		void drawGUI() override;

	private:
		std::size_t agentCounter{ 0 };
	};

}

#endif