#include "GUI/SparkGui.h"

#include "EngineSystems/ResourceManager.h"
#include "EngineSystems/SparkRenderer.h"
#include "ImGuizmo.h"
#include "ImGui/imgui_impl_glfw.h"
#include "JsonSerializer.h"
#include "Spark.h"
#include "ResourceLoader.h"

namespace spark {

	void SparkGui::drawGui()
	{
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGuizmo::BeginFrame();

		bool show = true;
		ImGui::ShowDemoWindow(&show);
		if (ImGui::BeginMainMenuBar())
		{
			drawMainMenuGui();
			SceneManager::getInstance()->drawGui();
			ResourceManager::getInstance()->drawGui();
			ImGui::EndMainMenuBar();
		}
	}

	void SparkGui::drawMainMenuGui()
	{
		static bool showEngineSettings = false;
		if (ImGui::BeginMenu("Engine"))
		{
			ImGui::MenuItem("Spark Settings", NULL, &showEngineSettings);
			ImGui::Separator();
			if (ImGui::MenuItem("Exit", "Esc"))
			{
				Spark::runProgram = false;
			}
			ImGui::EndMenu();
		}

		if (showEngineSettings)	drawSparkSettings(&showEngineSettings);
	}

	void SparkGui::drawSparkSettings(bool* p_open)
	{
		if (!ImGui::Begin("Spark Settings", p_open, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::End();
			return;
		}
		/*static char buf1[128];
		static char buf2[128];
		ImGui::InputTextWithHint("Path to Models", Spark::pathToModelMeshes.string().c_str(), buf1, 128);
		ImGui::InputTextWithHint("Path to Resources", Spark::pathToResources.string().c_str(), buf2, 128);*/

		ImGui::Text("Path to models:"); ImGui::SameLine(); ImGui::Text(Spark::pathToModelMeshes.string().c_str());
		ImGui::Text("Path to resources:"); ImGui::SameLine(); ImGui::Text(Spark::pathToResources.string().c_str());

		static const char* items[4] = { "1280x720", "1600x900", "1920x1080", "1920x1055" };
		static int current_item = checkCurrentItem(items);
		if (ImGui::Combo("Resolution", &current_item, items, IM_ARRAYSIZE(items)))
		{
			if (current_item == 0)
			{
				Spark::resizeWindow(1280, 720);
			}
			else if (current_item == 1)
			{
				Spark::resizeWindow(1600, 900);
			}
			else if (current_item == 2)
			{
				Spark::resizeWindow(1920, 1080);
			}
			else if (current_item == 3)
			{
				Spark::resizeWindow(1920, 1055);
			}
		}

        //TODO: fix this!
		//if (ImGui::Button("Save settings"))
		//{
		//	InitializationVariables variables;
		//	variables.width = Spark::WIDTH;
		//	variables.height = Spark::HEIGHT;
		//	variables.pathToResources = Spark::pathToResources;
		//	variables.pathToModels = Spark::pathToModelMeshes;
		//	JsonSerializer::writeToFile("settings.json", variables.serialize());
		//}
		ImGui::End();
	}

	int SparkGui::checkCurrentItem(const char** items) const
	{
		const std::string resolution = std::to_string(Spark::WIDTH) + "x" + std::to_string(Spark::HEIGHT);
		for (int i = 0; i < 4; i++)
		{
			std::string item(items[i]);
			if (item == resolution)
				return i;
		}
		return 0;
	}



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

	std::shared_ptr<PbrCubemapTexture> SparkGui::getCubemapTexture()
	{
		std::shared_ptr<PbrCubemapTexture> ptr = nullptr;
		if (ImGui::Button("Get CubemapTexture"))
		{
			ImGui::OpenPopup("Cubemap Textures");
		}

		if (ImGui::BeginPopupModal("Cubemap Textures", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
		{
			for (const auto& path : ResourceManager::getInstance()->getCubemapTexturePaths())
			{
				if (ImGui::Button(path.c_str()))
				{
					auto optional_ptr = ResourceLoader::loadHdrTexture(path);
					if(optional_ptr)
					{
						ptr = optional_ptr.value();
					}
					ImGui::CloseCurrentPopup();
				}
			}
			if (ImGui::Button("Close"))
			{
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}
		return ptr;
	}

	std::shared_ptr<Shader> SparkGui::getShader()
	{
		std::shared_ptr<Shader> ptr = nullptr;
		if (ImGui::Button("Get Shader"))
		{
			ImGui::OpenPopup("Shaders");
		}

		if (ImGui::BeginPopupModal("Shaders", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
		{
			for (const auto& shaderName : ResourceManager::getInstance()->getShaderNames())
			{
				if (ImGui::Button(shaderName.c_str()))
				{
					ptr = ResourceManager::getInstance()->getShader(shaderName);
					ImGui::CloseCurrentPopup();
				}
			}
			if (ImGui::Button("Close"))
			{
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}
		return ptr;
	}

}
