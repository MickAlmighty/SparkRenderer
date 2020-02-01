#include "AgentSpawner.h"

#include "ActorAI.h"
#include "EngineSystems/ResourceManager.h"
#include "GameObject.h"
#include "JsonSerializer.h"
#include "ModelMesh.h"
#include "Mesh.h"

namespace spark {

	SerializableType AgentSpawner::getSerializableType()
	{
		return SerializableType::SAgentSpawner;
	}

	Json::Value AgentSpawner::serialize()
	{
		Json::Value root;
		root["name"] = name;
		return root;
	}

	void AgentSpawner::deserialize(Json::Value& root)
	{
		name = root.get("name", "AgentSpawner").asString();
	}

	void AgentSpawner::update()
	{

	}

	void AgentSpawner::fixedUpdate()
	{

	}

	void AgentSpawner::drawGUI()
	{
		ImGui::Text("Agents Count: "); ImGui::SameLine(); ImGui::Text(std::to_string(agentCounter).c_str());
		static bool instancedRendering = true;
		if (ImGui::Checkbox("instanced rendering", &instancedRendering))
		{
			setInstancedRenderingForAgents(instancedRendering);
		}

		if (ImGui::Button("Add 1 actor"))
		{
			spawnAgents(1);
		}

		if (ImGui::Button("Add 10 actors"))
		{
			spawnAgents(10);
		}

		if (ImGui::Button("Add 64 actors"))
		{
			spawnAgents(64);
		}

		if (ImGui::Button("Add 1024 actors"))
		{
			spawnAgents(1024);
		}

		if (ImGui::Button("Add 20736 actors"))
		{
			spawnAgents(20736);
		}

		if (ImGui::Button("Add 65535 actors"))
		{
			spawnAgents(65535);
		}

		removeComponentGUI<AgentSpawner>();
	}

	void AgentSpawner::spawnAgents(int numberOfAgents)
	{
		const auto parent = getGameObject()->getParent();
		for (int i = 0; i < numberOfAgents; ++i)
		{
			auto gameObject = std::make_shared<GameObject>();
			const auto actorAiComponent = std::make_shared<ActorAI>();
			actorAiComponent->movementSpeed = 4.0f;
			actorAiComponent->autoWalking = true;
			gameObject->addComponent(actorAiComponent, gameObject);

			const auto modelMesh = std::make_shared<ModelMesh>();
			modelMesh->instanced = true;
			const auto pathsToModels = ResourceManager::getInstance()->getPathsToModels();
			const auto meshes = ResourceManager::getInstance()->findModelMeshes(pathsToModels[0]);
			modelMesh->setModel(std::make_pair(pathsToModels[0], meshes));
			gameObject->addComponent(modelMesh, gameObject);

			getGameObject()->getParent()->addChild(gameObject, parent);

			this->meshes.push_back(modelMesh);
		}
		agentCounter += numberOfAgents;
	}

	void AgentSpawner::setInstancedRenderingForAgents(bool mode) const
	{
		for (auto& modelMesh : meshes)
		{
			if(modelMesh.expired())
				continue;
			const auto model = modelMesh.lock();
			model->instanced = mode;
		}
	}

	AgentSpawner::AgentSpawner(std::string&& newName) : Component(newName)
	{

	}
}
