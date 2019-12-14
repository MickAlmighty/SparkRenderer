#include "TerrainGenerator.h"

#include "ActorAI.h"
#include "CUDA/DeviceMemory.h"
#include "CUDA/kernel.cuh"
#include "EngineSystems/ResourceManager.h"
#include "GameObject.h"
#include "JsonSerializer.h"
#include "ModelMesh.h"
#include "Mesh.h"

namespace spark {

	SerializableType TerrainGenerator::getSerializableType()
	{
		return SerializableType::STerrainGenerator;
	}

	Json::Value TerrainGenerator::serialize()
	{
		Json::Value root;
		root["name"] = name;
		return root;
	}

	void TerrainGenerator::deserialize(Json::Value& root)
	{
		name = root.get("name", "TerrainGenerator").asString();
	}

	void TerrainGenerator::update()
	{

	}

	void TerrainGenerator::fixedUpdate()
	{

	}

	void TerrainGenerator::drawGUI()
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

		removeComponentGUI<TerrainGenerator>();
	}

	TerrainGenerator::TerrainGenerator(std::string&& newName) : Component(newName)
	{

	}
}
