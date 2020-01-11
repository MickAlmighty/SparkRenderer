#include "AgentSpawner.h"

#include "ActorAI.h"
#include "CUDA/DeviceMemory.h"
#include "CUDA/kernel.cuh"
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

		if (ImGui::Button("Add 1 actor"))
		{
			const auto parent = getGameObject()->getParent();

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

			agentCounter += 1;
		}

		if (ImGui::Button("Add 10 actors"))
		{
			const auto parent = getGameObject()->getParent();
			for (int i = 0; i < 10; ++i)
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
			}
			agentCounter += 10;
		}

		if (ImGui::Button("Add 100 actors"))
		{
			const auto parent = getGameObject()->getParent();
			for (int i = 0; i < 100; ++i)
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
			}
			agentCounter += 100;
		}

		if (ImGui::Button("Add 1024 actors"))
		{
			const auto parent = getGameObject()->getParent();
			for (int i = 0; i < 1024; ++i)
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
			}
			agentCounter += 1024;
		}

		if (ImGui::Button("Add 16384 actors"))
		{
			const auto parent = getGameObject()->getParent();
			for (int i = 0; i < 1024 * 16; ++i)
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
			}
			agentCounter += 1024 * 16;
		}

		if (ImGui::Button("Add 65535 actors"))
		{
			const auto parent = getGameObject()->getParent();
			for (int i = 0; i < 65535; ++i)
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
			}
			agentCounter += 65535;
		}

		removeComponentGUI<AgentSpawner>();
	}

	AgentSpawner::AgentSpawner(std::string&& newName) : Component(newName)
	{

	}
}
