#include "TerrainGenerator.h"

#include <glm/gtc/noise.hpp>
#include <glm/gtc/random.hpp>
#include <stb_image/stb_image.h>

#include "CUDA/Agent.cuh"
#include "CUDA/DeviceMemory.h"
#include "CUDA/kernel.cuh"
#include "EngineSystems/ResourceManager.h"
#include "GameObject.h"
#include "JsonSerializer.h"
#include "MeshPlane.h"
#include "Timer.h"

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
	
	static cudaStream_t stream1, stream2;
	if (ImGui::Button("InitMap"))
	{
		using namespace cuda;
		const auto nodes = DeviceMemory<float>::AllocateElements(terrainSize * terrainSize);
		gpuErrchk(cudaGetLastError());
		cudaMemcpy(nodes.ptr, terrain.data(), sizeof(float) * terrainSize * terrainSize, cudaMemcpyHostToDevice);
		gpuErrchk(cudaGetLastError());

		initMap(nodes.ptr, terrainSize, terrainSize);

		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);
	}

	if (ImGui::Button("FindPath"))
	{
		using namespace cuda;
		//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);
		//gpuErrchk(cudaGetLastError());
		Timer timer1("CUDA");

		glm::ivec2 path[] = { {10, 10}, {15, 10}};
		/*const auto pathDev = DeviceMemory<int>::AllocateBytes(sizeof(glm::ivec2) * 2);
		gpuErrchk(cudaGetLastError());
		cudaMemcpyAsync(pathDev.ptr, &path, sizeof(glm::ivec2) * 2, cudaMemcpyHostToDevice, stream1);
		gpuErrchk(cudaGetLastError());
		
		const auto agents = DeviceMemory<Agent>::AllocateElements(1);
		gpuErrchk(cudaGetLastError());
		
		const auto memSize = DeviceMemory<int>::AllocateElements(1);
		gpuErrchk(cudaGetLastError());*/

		const std::size_t byteCount = sizeof(glm::ivec2) * 2 + sizeof(Agent) + sizeof(int);
		const int pathOffset = 0;
		const int agentsOffset = sizeof(glm::ivec2) * 2;
		const int memSizeOffset = agentsOffset + sizeof(int);

		const auto devMem = DeviceMemory<char>::AllocateBytes(byteCount);
		gpuErrchk(cudaGetLastError());

		cudaMemcpyAsync(devMem.ptr, &path, sizeof(glm::ivec2) * 2, cudaMemcpyHostToDevice, stream1);
		gpuErrchk(cudaGetLastError());

		const auto pathDev = reinterpret_cast<int*>(devMem.ptr + pathOffset);
		const auto agentsDev = reinterpret_cast<Agent*>(devMem.ptr + agentsOffset);
		const auto memSizeDev = reinterpret_cast<int*>(devMem.ptr + memSizeOffset);
		runKernel(pathDev, memSizeDev, agentsDev);
		//cudaStreamDestroy(stream1);
		//cudaStreamDestroy(stream2);
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

int TerrainGenerator::getTerrainNodeIndex(const int x, const int y) const
{
	return y * terrainSize + x;
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

float TerrainGenerator::getTerrainValue(const int x, const int y)
{
	const unsigned int index = getTerrainNodeIndex(x, y);
	return terrain[index];
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
