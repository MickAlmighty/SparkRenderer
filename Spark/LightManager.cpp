#include "LightManager.h"

#include "DirectionalLight.h"
#include "PointLight.h"
#include "SpotLight.h"
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

	void LightManager::addSpotLight(const std::shared_ptr<SpotLight>& spotLight)
	{
		spotLights.push_back(spotLight);
	}

	void LightManager::updateLightBuffers()
	{
		const auto dirLightDataBuffer = getLightDataBuffer<DirectionalLightData, DirectionalLight>(directionalLights);
		updateBufferIfNecessary(dirLightDataBuffer, dirLightSSBO);

		const auto pointLightDataBuffer = getLightDataBuffer<PointLightData, PointLight>(pointLights);
		updateBufferIfNecessary(pointLightDataBuffer, pointLightSSBO);

		const auto spotLightDataBuffer = getLightDataBuffer<SpotLightData, SpotLight>(spotLights);
		updateBufferIfNecessary(spotLightDataBuffer, spotLightSSBO);
	}

	LightManager::LightManager()
	{
		glGenBuffers(1, &dirLightSSBO);
		glGenBuffers(1, &pointLightSSBO);
		glGenBuffers(1, &spotLightSSBO);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, dirLightSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, pointLightSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, spotLightSSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	LightManager::~LightManager()
	{
		glDeleteBuffers(1, &dirLightSSBO);
		glDeleteBuffers(1, &pointLightSSBO);
		glDeleteBuffers(1, &spotLightSSBO);
	}

std::vector<std::shared_ptr<DirectionalLight>> LightManager::getDirectionalLights() {
    std::vector<std::shared_ptr<DirectionalLight>> vec;
    for(auto& light : directionalLights)
    {
        vec.push_back(light.lock());
    }
    return vec;
}

std::vector<std::shared_ptr<PointLight>> LightManager::getPointLights()
{
    std::vector<std::shared_ptr<PointLight>> vec;
    for(auto& light : pointLights)
    {
        vec.push_back(light.lock());
    }
    return vec;
}

std::vector<std::shared_ptr<SpotLight>> LightManager::getSpotLights()
{
    std::vector<std::shared_ptr<SpotLight>> vec;
    for(auto& light : spotLights)
    {
        vec.push_back(light.lock());
    }
    return vec;
}

void LightManager::setDirectionalLights(std::vector<std::shared_ptr<DirectionalLight>> lights) {
    directionalLights.clear();
    for(auto& light : lights)
    {
        directionalLights.push_back(light);
    }
}

void LightManager::setPointLights(std::vector<std::shared_ptr<PointLight>> lights)
{
    pointLights.clear();
    for(auto& light : lights)
    {
        pointLights.push_back(light);
    }
}

void LightManager::setSpotLights(std::vector<std::shared_ptr<SpotLight>> lights)
{
    spotLights.clear();
    for(auto& light : lights)
    {
        spotLights.push_back(light);
    }
}
}

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::LightManager>("LightManager")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("directionalLights", &spark::LightManager::getDirectionalLights, &spark::LightManager::setDirectionalLights,
                  rttr::registration::public_access)
        .property("pointLights", &spark::LightManager::getPointLights, &spark::LightManager::setPointLights, rttr::registration::public_access)
        .property("spotLights", &spark::LightManager::getSpotLights, &spark::LightManager::setSpotLights, rttr::registration::public_access);
}