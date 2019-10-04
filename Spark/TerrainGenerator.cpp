#include "TerrainGenerator.h"

#include <glm/gtc/noise.hpp>
#include <glm/gtc/random.hpp>

#include "EngineSystems/ResourceManager.h"
#include "GameObject.h"
#include "MeshPlane.h"

namespace spark {

std::vector<glm::vec3> TerrainGenerator::getPerlinValues() const
{
	return perlinValues;
}

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
	Texture tex = generateTerrain();
	ResourceManager::getInstance()->addTexture(tex);
}

void TerrainGenerator::update()
{

}

void TerrainGenerator::fixedUpdate()
{

}

void TerrainGenerator::drawGUI()
{
	ImGui::PushID(this);
	ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
	ImGui::SetNextWindowSizeConstraints(ImVec2(250, 0), ImVec2(FLT_MAX, 150)); // Width = 250, Height > 100
	ImGui::BeginChild("TerrainGenerator", { 0, 0 }, true, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_AlwaysAutoResize);
	if (ImGui::BeginMenuBar())
	{
		ImGui::Text("TerrainGenerator");
		ImGui::EndMenuBar();
	}

	ImGui::DragInt("TerrainSize", &terrainSize, 1);
	ImGui::DragFloat("PerlinDivider", &perlinDivider, 0.003f);
	ImGui::DragFloat("PerlinTimeStep", &perlinTimeStep, 0.1f);

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

	ImGui::EndChild();
	ImGui::PopStyleVar();
	ImGui::PopID();
}

Texture TerrainGenerator::generateTerrain()
{
	const int width = terrainSize, height = terrainSize;
	perlinValues.clear();
	float y = glm::linearRand(20.0f, 100.0f);
	float x = y;
	for (int i = 0; i < width; ++i)
	{
		for (int j = 0; j < height; ++j)
		{
			float perlinValue = glm::perlin(glm::vec2(x / perlinDivider, y / perlinDivider));
			glm::vec3 perlinNoise{ glm::clamp(perlinValue, 0.0f, 1.0f), 0, 0 };
			perlinValues.push_back(perlinNoise);
			y += perlinTimeStep;
		}
		x += perlinTimeStep;
	}

	updateTerrain(perlinValues);
	generatedTerrain.path = "GeneratedTerrain";
	return generatedTerrain;
}

void TerrainGenerator::updateTerrain(std::vector<glm::vec3> newPerlinValues) const
{
	glBindTexture(GL_TEXTURE_2D, generatedTerrain.ID);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, perlinValues.data());
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, terrainSize, terrainSize, GL_RGB, GL_FLOAT, newPerlinValues.data());
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
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, terrainSize, terrainSize, 0, GL_RGB, GL_FLOAT, NULL);
}


TerrainGenerator::~TerrainGenerator()
{
	glDeleteTextures(1, &generatedTerrain.ID);
}
}