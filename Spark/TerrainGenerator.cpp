#include <TerrainGenerator.h>
#include <GameObject.h>
#include <MeshPlane.h>
#include <glm/gtc/noise.hpp>
#include <glm/gtc/random.hpp>

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

	if(ImGui::Button("Generate Terrain"))
	{
		Texture tex = generateTerrain();
		auto meshPlane = getGameObject()->getComponent<MeshPlane>();
		if(meshPlane != nullptr)
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
	float y = glm::linearRand(0.0f, 100.0f);
	float x = y;
	for(int i = 0; i < width; ++i)
	{
		for(int j = 0; j < height; ++j)
		{
			perlinValues.push_back(glm::clamp(glm::perlin(glm::vec2(x / perlinDivider, y / perlinDivider)), 0.0f, 1.0f));
			y += perlinTimeStep;
		}
		x += perlinTimeStep;
	}

	GLuint id;
	glGenTextures(1, &id);
	glBindTexture(GL_TEXTURE_2D, id);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_FLOAT, perlinValues.data());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
	Texture t{id, ""};

	return t;
}


TerrainGenerator::TerrainGenerator(std::string&& newName) : Component(newName)
{

}


TerrainGenerator::~TerrainGenerator()
{
}


