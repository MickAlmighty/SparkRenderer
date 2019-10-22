#include "LightManager.h"

#include "DirectionalLight.h"
#include "PointLight.h"
#include "Structs.h"

namespace spark {

	void LightManager::addDirectionalLight(const std::shared_ptr<DirectionalLight>& directionalLight)
	{
		directionalLights.push_back(directionalLight);
	}

	void LightManager::addPointLight(const std::shared_ptr<PointLight>& pointLight)
	{
		pointLights.push_back(pointLight);
	}

	void LightManager::updateLightBuffers()
	{
		const auto dirLightDataBuffer = getLightDataBuffer<DirectionalLightData, DirectionalLight>(directionalLights);
		updateBufferIfNecessary(dirLightDataBuffer, dirLightSSBO);

		const auto pointLightDataBuffer = getLightDataBuffer<PointLightData, PointLight>(pointLights);
		updateBufferIfNecessary(pointLightDataBuffer, pointLightSSBO);
	}

	LightManager::LightManager()
	{
		glGenBuffers(1, &dirLightSSBO);
		glGenBuffers(1, &pointLightSSBO);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirLightSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, pointLightSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	LightManager::~LightManager()
	{
		glDeleteBuffers(1, &dirLightSSBO);
		glDeleteBuffers(1, &pointLightSSBO);
	}

}

