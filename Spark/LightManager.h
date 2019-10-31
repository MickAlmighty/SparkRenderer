#ifndef LIGHT_MANAGER_H
#define LIGHT_MANAGER_H

#include <algorithm>
#include <memory>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace spark {
	class DirectionalLight;
	class PointLight;
	class SpotLight;

	class LightManager
	{
	public:
		GLuint dirLightSSBO{}, pointLightSSBO{}, spotLightSSBO{};

		void addDirectionalLight(const std::shared_ptr<DirectionalLight>& directionalLight);
		void addPointLight(const std::shared_ptr<PointLight>& pointLight);
		void addSpotLight(const std::shared_ptr<SpotLight>& spotLight);
		void updateLightBuffers();
		
		LightManager();
		~LightManager();
		LightManager(const LightManager& lightManager) = delete;
		LightManager(const LightManager&& lightManager) = delete;
		LightManager& operator=(const LightManager& lightManager) = delete;
		LightManager&& operator=(const LightManager&& lightManager) = delete;
	

	private:
		std::vector<std::weak_ptr<DirectionalLight>> directionalLights;
		std::vector<std::weak_ptr<PointLight>> pointLights;
		std::vector<std::weak_ptr<SpotLight>> spotLights;
		bool updateBuffer = false;

		template <typename T>
		void updateBufferIfNecessary(const std::vector<T>& bufferLightData, GLuint ssbo)
		{
			if (updateBuffer)
			{
				const GLuint size = sizeof(T);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				glBufferData(GL_SHADER_STORAGE_BUFFER, bufferLightData.size() * size, bufferLightData.data(), GL_DYNAMIC_DRAW);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
				updateBuffer = false;
			}
		}

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
				if(expiredPointer)
				{
					updateBuffer = true;
				}
			} while (expiredPointer);


			if (updateBuffer)
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