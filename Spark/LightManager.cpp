#include "LightManager.h"

#include <algorithm>

#include "DirectionalLight.h"

namespace spark {

	void LightManager::addDirectionalLight(const std::shared_ptr<DirectionalLight>& directionalLight)
	{
		directionalLights.push_back(directionalLight);
	}

	void LightManager::updateLightBuffers()
	{

	}
}

