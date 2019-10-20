#include "LightManager.h"

#include "DirectionalLight.h"
#include "Structs.h"

namespace spark {

	void LightManager::addDirectionalLight(const std::shared_ptr<DirectionalLight>& directionalLight)
	{
		directionalLights.push_back(directionalLight);
	}

	void LightManager::updateLightBuffers()
	{
		const auto lightDataBuffer = getLightDataBuffer<DirectionalLightData, DirectionalLight>(directionalLights);

		if (!lightDataBuffer.empty())
		{
			float f = 5;
			//#todo: update buffer
		}
	}

}

