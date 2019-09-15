#include <GUI/SparkGui.h>
#include "EngineSystems/SparkRenderer.h"
#include "JsonSerializer.h"

void SparkGui::drawMainMenuGui()
{
	static bool showEngineSettings = false;
	if (ImGui::BeginMainMenuBar())
	{
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
		if (ImGui::BeginMenu("SceneManager"))
		{
			//ImGui::Text("Current Scene:"); ImGui::SameLine(); ImGui::Text()
			ImGui::MenuItem("Spark Settings", NULL, &showEngineSettings);
			ImGui::Separator();
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	if(showEngineSettings)	drawSparkSettings(&showEngineSettings);
}

static int checkCurrentItem(const char** items)
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

void SparkGui::drawSparkSettings(bool *p_open)
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

	static const char* items[4] = {"1280x720", "1600x900", "1920x1080", "1920x1055" };
	static int current_item = checkCurrentItem(items);
	if(ImGui::Combo("Resolution", &current_item, items, IM_ARRAYSIZE(items)))
	{
		if(current_item == 0)
		{
			SparkRenderer::resizeWindow(1280, 720);
		}
		else if (current_item == 1)
		{
			SparkRenderer::resizeWindow(1600, 900);
		}
		else if (current_item == 2)
		{
			SparkRenderer::resizeWindow(1920, 1080);
		}
		else if (current_item == 3)
		{
			SparkRenderer::resizeWindow(1920, 1055);
		}
	}

	if (ImGui::Button("Save settings"))
	{
		InitializationVariables variables;
		variables.width = Spark::WIDTH;
		variables.height = Spark::HEIGHT;
		variables.pathToResources = Spark::pathToResources;
		variables.pathToModels = Spark::pathToModelMeshes;
		JsonSerializer::writeToFile("settings.json", variables.serialize());
	}
	ImGui::End();
}


