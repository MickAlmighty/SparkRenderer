#pragma once

#include <algorithm>
#include <memory>
#include <vector>

namespace spark {
	class DirectionalLight;
	
	class LightManager
	{
	public:
		void addDirectionalLight(const std::shared_ptr<DirectionalLight>& directionalLight);
		void updateLightBuffers();


		LightManager() = default;
		~LightManager() = default;
		LightManager(const LightManager& lightManager) = delete;
		LightManager(const LightManager&& lightManager) = delete;
		LightManager& operator=(const LightManager& lightManager) = delete;
		LightManager&& operator=(const LightManager&& lightManager) = delete;
	

	private:
		std::vector<std::weak_ptr<DirectionalLight>> directionalLights;

		template <typename T>
		bool findExpiredPointer(std::vector<std::weak_ptr<T>>& lightContainer)
		{
			const auto containerIt = std::find_if(std::begin(lightContainer), std::end(lightContainer), [](const std::weak_ptr<T>& weakPtr)
			{
				return weakPtr.expired();
			});

			if(containerIt != std::end(lightContainer))
			{
				lightContainer.erase(containerIt);
				return true;
			}
			return false;
		}
	};
}


