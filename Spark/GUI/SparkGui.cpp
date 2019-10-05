#include "GUI/SparkGui.h"

#include "JsonSerializer.h"
#include "EngineSystems/ResourceManager.h"
#include "EngineSystems/SparkRenderer.h"

namespace spark {

std::shared_ptr<Component> SparkGui::addComponent()
{
	std::shared_ptr<Component> component = nullptr;
	if (ImGui::Button("Add Component"))
	{
		ImGui::OpenPopup("Components");
	}

	if (ImGui::BeginPopupModal("Components", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		for (const auto& componentType : componentCreation)
		{
			if (ImGui::Button(componentType.first.c_str()))
			{
				component = componentType.second();
				ImGui::CloseCurrentPopup();
			}
		}
		if (ImGui::Button("Close"))
		{
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}
	return component;
}

std::pair<std::string, std::vector<Mesh>> SparkGui::getMeshes()
{
	std::pair<std::string, std::vector<Mesh>> meshes;
	if (ImGui::Button("Add Model Meshes"))
	{
		ImGui::OpenPopup("Models");
	}

	if (ImGui::BeginPopupModal("Models", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		std::vector<std::string> pathsToModels = ResourceManager::getInstance()->getPathsToModels();
		for (const std::string& path : pathsToModels)
		{
			if (ImGui::Button(path.c_str()))
			{
				meshes.first = path;
				meshes.second = ResourceManager::getInstance()->findModelMeshes(path);
				ImGui::CloseCurrentPopup();
			}
		}
		if (ImGui::Button("Close"))
		{
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}
	return meshes;
}

Texture SparkGui::getTexture()
{
	Texture tex{ 0, "" };
	if (ImGui::Button("Add Texture"))
	{
		ImGui::OpenPopup("Textures");
	}

	if (ImGui::BeginPopupModal("Textures", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		for (const Texture& texture : ResourceManager::getInstance()->getTextures())
		{
			if (ImGui::Button(texture.path.c_str()))
			{
				tex = texture;
				ImGui::CloseCurrentPopup();
			}
		}
		if (ImGui::Button("Close"))
		{
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}
	
	return tex;
}

}