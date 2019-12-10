#include "TerrainGenerator.h"

#include <glm/gtc/noise.hpp>
#include <glm/gtc/random.hpp>
#include <stb_image/stb_image.h>

#include "ActorAI.h"
#include "CUDA/DeviceMemory.h"
#include "CUDA/kernel.cuh"
#include "EngineSystems/ResourceManager.h"
#include "GameObject.h"
#include "JsonSerializer.h"
#include "Mesh.h"
#include "MeshPlane.h"
#include "ModelMesh.h"

namespace spark {

SerializableType TerrainGenerator::getSerializableType()
{
	return SerializableType::STerrainGenerator;
}

Json::Value TerrainGenerator::serialize()
{
	Json::Value root;
	root["name"] = name;
	root["terrainSize"] = terrainSize;
	return root;
}

void TerrainGenerator::deserialize(Json::Value& root)
{
	name = root.get("name", "TerrainGenerator").asString();
	terrainSize = root.get("terrainSize", 20).asInt();
	
	int tex_width, tex_height, nr_channels;
	
	unsigned char* pixels = stbi_load("map.png", &tex_width, &tex_height, &nr_channels, 0);

	std::vector<glm::vec3> pix;
	pix.reserve(tex_width * tex_height);
	for (int i = 0; i < tex_width * tex_height * nr_channels; i += 3)
	{
		glm::vec3 pixel;
		pixel.x = *(pixels + i);
		pixel.y = *(pixels + i + 1);
		pixel.z = *(pixels + i + 2);
		if(pixel != glm::vec3(0))
			pix.push_back(glm::normalize(pixel));
		else
			pix.push_back(pixel);
	}
//#TODO: Terrain size as rectangle(x,y) not quad(x,x) 
	terrainSize = tex_width;
	terrain = std::vector<float>(tex_width * tex_height);
	for (int i = 0; i < terrainSize * terrainSize; i++)
	{
		terrain[i] = pix[i].x;
	}
	updateTerrain();
	generatedTerrain.path = "GeneratedTerrain";
	ResourceManager::getInstance()->addTexture(generatedTerrain);
}

void TerrainGenerator::update()
{

}

void TerrainGenerator::fixedUpdate()
{

}

void TerrainGenerator::drawGUI()
{
	ImGui::Text("TerrainSize: "); ImGui::SameLine(); ImGui::Text(std::to_string(terrainSize).c_str());
	ImGui::Text("Agents Count: "); ImGui::SameLine(); ImGui::Text(std::to_string(agentCounter).c_str());
	
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

	if(ImGui::Button("Map from .bmp"))
	{
		int tex_width, tex_height, nr_channels;
		unsigned char* pixels = stbi_load("map.png", &tex_width, &tex_height, &nr_channels, 0);

		std::vector<glm::vec3> pix;
		pix.reserve(tex_width * tex_height);
		for(int i = 0; i < tex_width * tex_height * nr_channels; i += 3)
		{
			glm::vec3 pixel;
			pixel.x = *(pixels + i);
			pixel.y = *(pixels + i + 1);
			pixel.z = *(pixels + i + 2);
			if (pixel != glm::vec3(0))
				pix.push_back(glm::normalize(pixel));
			else
				pix.push_back(pixel);
		}

		for(int i = 0; i < terrainSize * terrainSize; i++)
		{
			terrain[i] = pix[i].x;
		}
		updateTerrain();
	}
	
	if (ImGui::Button("Generate Terrain"))
	{
		Texture tex = generateTerrain();
		auto meshPlane = getGameObject()->getComponent<MeshPlane>();
		if (meshPlane != nullptr)
		{
			meshPlane->setTexture(TextureTarget::DIFFUSE_TARGET, tex);
		}
	}

	removeComponentGUI<TerrainGenerator>();
}

Texture TerrainGenerator::generateTerrain()
{
	updateTerrain();
	generatedTerrain.path = "GeneratedTerrain";
	return generatedTerrain;
}

void TerrainGenerator::updateTerrain() const
{
	glBindTexture(GL_TEXTURE_2D, generatedTerrain.ID);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, perlinValues.data());
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, terrainSize, terrainSize, GL_RED, GL_FLOAT, terrain.data());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}

TerrainGenerator::TerrainGenerator(std::string&& newName) : Component(newName)
{
	glGenTextures(1, &generatedTerrain.ID);
	glBindTexture(GL_TEXTURE_2D, generatedTerrain.ID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, terrainSize, terrainSize, 0, GL_RED, GL_FLOAT, NULL);
}


TerrainGenerator::~TerrainGenerator()
{
	glDeleteTextures(1, &generatedTerrain.ID);
}
}
