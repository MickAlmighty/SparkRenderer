#include "GUI/SparkGui.h"

#include "JsonSerializer.h"
#include "EngineSystems/ResourceManager.h"
#include "EngineSystems/SparkRenderer.h"

namespace spark {

std::shared_ptr<Component> SparkGui::addComponent()
{
	static bool componentWindowAddition = false;
	std::shared_ptr<Component> component = nullptr;
	if (ImGui::Button("Add Component"))
	{
		componentWindowAddition = true;
	}

	if (componentWindowAddition)
	{

		if (ImGui::Begin("Components", &componentWindowAddition))
		{
			for (const auto& componentType : componentCreation)
			{
				if (ImGui::Button(componentType.first.c_str()))
				{
					component = componentType.second();
					componentWindowAddition = false;
				}
			}
			if (ImGui::Button("Close"))
			{
				componentWindowAddition = false;
			}
		}
		ImGui::End();
	}
	return component;
}

std::pair<std::string, std::vector<Mesh>> SparkGui::getMeshes()
{
	static bool componentWindowAddition = false;
	std::pair<std::string, std::vector<Mesh>> meshes;
	if (ImGui::Button("Add Model Meshes"))
	{
		componentWindowAddition = true;
	}

	if (componentWindowAddition)
	{
		if (ImGui::Begin("Models", &componentWindowAddition))
		{
			std::vector<std::string> pathsToModels = ResourceManager::getInstance()->getPathsToModels();
			for (const std::string& path : pathsToModels)
			{
				if (ImGui::Button(path.c_str()))
				{
					meshes.first = path;
					meshes.second = ResourceManager::getInstance()->findModelMeshes(path);
					componentWindowAddition = false;
				}
			}
			if (ImGui::Button("Close"))
			{
				componentWindowAddition = false;
			}
		}
		ImGui::End();
	}
	return meshes;
}

Texture SparkGui::getTexture()
{
	static bool textureWindow = false;
	Texture tex{ 0, "" };
	if (ImGui::Button("Add Texture"))
	{
		textureWindow = true;
	}

	if (textureWindow)
	{

		if (ImGui::Begin("Textures", &textureWindow))
		{
			for (const Texture& texture : ResourceManager::getInstance()->getTextures())
			{
				if (ImGui::Button(texture.path.c_str()))
				{
					tex = texture;
					textureWindow = false;
				}
			}
			if (ImGui::Button("Close"))
			{
				textureWindow = false;
			}
		}
		ImGui::End();
	}
	return tex;
}

}