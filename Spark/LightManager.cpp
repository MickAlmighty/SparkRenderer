#include "LightManager.h"

#include "DirectionalLight.h"
#include "Structs.h"
#include <iostream>

namespace spark {

	void LightManager::addDirectionalLight(const std::shared_ptr<DirectionalLight>& directionalLight)
	{
		directionalLights.push_back(directionalLight);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirLightSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	void LightManager::updateLightBuffers()
	{
		const auto lightDataBuffer = getLightDataBuffer<DirectionalLightData, DirectionalLight>(directionalLights);

		if (updateBuffer)
		{
			//std::cout << "SSBO update!\n";
			const GLuint size = sizeof(DirectionalLightData);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirLightSSBO);
			glBufferData(GL_SHADER_STORAGE_BUFFER, lightDataBuffer.size() * size, lightDataBuffer.data(), GL_DYNAMIC_DRAW);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, dirLightSSBO);
			updateBuffer = false;
		}
	}

	LightManager::LightManager()
	{
		glGenBuffers(1, &dirLightSSBO);
	}

	LightManager::~LightManager()
	{
		glDeleteBuffers(1, &dirLightSSBO);
	}

}

