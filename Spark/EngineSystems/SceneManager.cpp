#include <EngineSystems/SceneManager.h>
#include <EngineSystems/SparkRenderer.h>
#include <GUI/ImGui/imgui.h>
#include <Spark.h>
#include <JsonSerializer.h>
#include <algorithm>
#include <Scene.h>


std::shared_ptr<SceneManager> SceneManager::getInstance()
{
	static std::shared_ptr<SceneManager> scene_manager = nullptr;
	if(scene_manager == nullptr)
	{
		scene_manager = std::make_shared<SceneManager>();
	}
	return scene_manager;
}

void SceneManager::setup()
{
	auto scene = std::make_shared<Scene>("MainScene");
	addScene(scene);
	setCurrentScene("MainScene");

}

void SceneManager::update() const
{
	current_scene->update();
}

void SceneManager::fixedUpdate() const
{
	current_scene->fixedUpdate();
}

void SceneManager::cleanup()
{
	current_scene = nullptr;
	scenes.clear();
}

void SceneManager::addScene(const std::shared_ptr<Scene>& scene)
{
	scenes.push_back(scene);
}

bool SceneManager::setCurrentScene(std::string&& sceneName)
{
	const auto searchingFunction = [&sceneName](const std::shared_ptr<Scene>& scene)
	{
		return scene->name == sceneName;
	};

	const auto scene_it = std::find_if(std::begin(scenes), std::end(scenes), searchingFunction);
	
	if(scene_it != std::end(scenes))
	{
		current_scene = *scene_it;
		return true;
	}
	return false;
}

std::shared_ptr<Scene> SceneManager::getCurrentScene()
{
	return current_scene;
}

void SceneManager::drawGUI()
{
	bool show = true;
	ImGui::ShowDemoWindow(&show);

	drawMainMenuGui();

	current_scene->drawGUI();
}

void SceneManager::drawMainMenuGui()
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
			ImGui::Text("Current Scene:"); ImGui::SameLine(); ImGui::Text(current_scene->name.c_str());
			if(ImGui::Button("Save Current Scene"))
			{
				JsonSerializer::writeToFile("scene.json",current_scene->serialize());
			}
			if (ImGui::Button("Load main scene"))
			{
				Json::Value root = JsonSerializer::readFromFile("scene.json");
				current_scene->deserialize(root);
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	if (showEngineSettings)	drawSparkSettings(&showEngineSettings);
}

void SceneManager::drawSparkSettings(bool* p_open)
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

int SceneManager::checkCurrentItem(const char** items) const
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