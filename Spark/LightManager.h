#ifndef LIGHT_MANAGER_H
#define LIGHT_MANAGER_H

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
		std::vector<std::weak_ptr<T>> findDirtyLights(const std::vector<std::weak_ptr<T>>& lightContainer)
		{
			std::vector<std::weak_ptr<T>> dirtyLights;
			for(const std::weak_ptr<T>& light : lightContainer)
			{
				if(light.lock()->getActive())
				{
					dirtyLights.push_back(light);
				}
			}
			return dirtyLights;
		}

		template <typename T>
		bool findAndRemoveExpiredPointer(std::vector<std::weak_ptr<T>>& lightContainer)
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

		template <typename N, typename T>
		std::vector<N> getLightDataBuffer(std::vector<std::weak_ptr<T>>& lightContainer)
		{
			bool expiredPointer = false;
			do {
				expiredPointer = findAndRemoveExpiredPointer(lightContainer);
			} while (expiredPointer);


			if (expiredPointer)
			{
				std::vector<N> bufferData;
				for (const auto& light : lightContainer)
				{
					bufferData.push_back(light.lock()->getLightData());
				}

				return bufferData;
			}
			else
			{
				std::vector<N> bufferData;
				const auto dirtyLights = findDirtyLights(lightContainer);
				for (const auto& light : dirtyLights)
				{
					bufferData.push_back(light.lock()->getLightData());
				}

				return bufferData;
			}
		}
	};
}


#endif