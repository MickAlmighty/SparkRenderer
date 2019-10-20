#ifndef LIGHT_MANAGER_H
#define LIGHT_MANAGER_H

#include <algorithm>
#include <memory>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace spark {
	class DirectionalLight;

	class LightManager
	{
	public:
		GLuint dirLightSSBO{};

		void addDirectionalLight(const std::shared_ptr<DirectionalLight>& directionalLight);
		void updateLightBuffers();
		
		LightManager();
		~LightManager();
		LightManager(const LightManager& lightManager) = delete;
		LightManager(const LightManager&& lightManager) = delete;
		LightManager& operator=(const LightManager& lightManager) = delete;
		LightManager&& operator=(const LightManager&& lightManager) = delete;
	

	private:
		std::vector<std::weak_ptr<DirectionalLight>> directionalLights;
		bool updateBuffer = false;

		template <typename T>
		bool findDirtyLight(const std::vector<std::weak_ptr<T>>& lightContainer)
		{
			for(const std::weak_ptr<T>& light : lightContainer)
			{
				if(light.lock()->getDirty())
				{
					return true;
				}
			}
			return false;
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
				updateBuffer = findDirtyLight(lightContainer);
				if (updateBuffer)
				{
					for (const auto& light : lightContainer)
					{
						light.lock()->resetDirty();
						if (light.lock()->getActive())
						{
							bufferData.push_back(light.lock()->getLightData());
						}
					}
				}
				return bufferData;
			}
		}
	};
}


#endif