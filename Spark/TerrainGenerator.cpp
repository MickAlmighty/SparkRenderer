#include "TerrainGenerator.h"

#include <glm/gtc/noise.hpp>
#include <glm/gtc/random.hpp>
#include <stb_image/stb_image.h>

#include "EngineSystems/ResourceManager.h"
#include "GameObject.h"
#include "MeshPlane.h"
#include "JsonSerializer.h"

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
	unsigned char* pixels;
	pixels = stbi_load("map.png", &tex_width, &tex_height, &nr_channels, 0);

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

	terrain = std::vector<TerrainNode>(400);
	for (int i = 0; i < terrainSize * terrainSize; i++)
	{
		terrain[i].nodeData.x = pix[i].x;
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

	if(ImGui::Button("Map from .bmp"))
	{
		int tex_width, tex_height, nr_channels;
		unsigned char* pixels;
		pixels = stbi_load("map.png", &tex_width, &tex_height, &nr_channels, 0);

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
			terrain[i].nodeData.x = pix[i].x;
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

	ImGui::EndChild();
	ImGui::PopStyleVar();
	ImGui::PopID();
}

int TerrainGenerator::getTerrainNodeIndex(const int x, const int y) const
{
	return y * terrainSize + x;
}

Texture TerrainGenerator::generateTerrain()
{
	const int width = terrainSize, height = terrainSize;
	terrain.clear();
	float y = glm::linearRand(20.0f, 100.0f);
	float x = y;
	for (int i = 0; i < width; ++i)
	{
		for (int j = 0; j < height; ++j)
		{
			const float perlinValue = glm::perlin(glm::vec2(x / perlinDivider, y / perlinDivider));
			const glm::vec3 perlinNoise{ glm::clamp(perlinValue, 0.0f, 1.0f), 0, 0 };
			terrain.push_back(TerrainNode{ 0, perlinNoise });
			y += perlinTimeStep;
		}
		x += perlinTimeStep;
	}

	updateTerrain();
	generatedTerrain.path = "GeneratedTerrain";
	return generatedTerrain;
}

void TerrainGenerator::updateTerrain() const
{
	std::vector<glm::vec3> pixels;
	for(const auto terrainNode : terrain)
	{
		pixels.push_back(terrainNode.nodeData);
	}
	glBindTexture(GL_TEXTURE_2D, generatedTerrain.ID);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, perlinValues.data());
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, terrainSize, terrainSize, GL_RGB, GL_FLOAT, pixels.data());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void TerrainGenerator::markNodeAsPartOfPath(int x, int y)
{
	terrain[getTerrainNodeIndex(x, y)].nodeData.y = 1.0f;
	terrain[getTerrainNodeIndex(x, y)].numberOfActorsPassingThrough++;
}

void TerrainGenerator::unMarkNodeAsPartOfPath(int x, int y)
{
	terrain[getTerrainNodeIndex(x, y)].numberOfActorsPassingThrough--;
	if(terrain[getTerrainNodeIndex(x, y)].numberOfActorsPassingThrough == 0)
	{
		terrain[getTerrainNodeIndex(x, y)].nodeData.y = 0.0f;
	}
}

float TerrainGenerator::getTerrainValue(const int x, const int y)
{
	const unsigned int index = getTerrainNodeIndex(x, y);
	return terrain[index].nodeData.x;
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
