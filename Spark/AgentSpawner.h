#ifndef TERRAIN_GENERATOR_H
#define TERRAIN_GENERATOR_H

#include "Component.h"

namespace spark {
	class ModelMesh;

	class AgentSpawner : public Component
	{
	public:
		AgentSpawner(std::string&& newName = "AgentSpawner");
		~AgentSpawner() = default;

		SerializableType getSerializableType() override;
		Json::Value serialize() override;
		void deserialize(Json::Value& root) override;
		void update() override;
		void fixedUpdate() override;
		void drawGUI() override;

	private:
		std::size_t agentCounter{ 0 };
		std::vector<std::weak_ptr<ModelMesh>> meshes;

		void spawnAgents(int numberOfAgents);
		void setInstancedRenderingForAgents(bool mode) const;
	};

}

#endif