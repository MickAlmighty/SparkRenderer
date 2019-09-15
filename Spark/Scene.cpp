#include <Scene.h>
#include <list>
#include <GUI/ImGui/imgui.h>

Scene::Scene(std::string&& sceneName) : name(sceneName)
{

}

Scene::~Scene()
{
#ifdef DEBUG
	std::cout << "Scene destroyed!" << std::endl;
#endif
}

void Scene::update() const
{
	root->update();
}

void Scene::fixedUpdate() const
{
	root->fixedUpdate();
}

void Scene::removeGameObject(std::string&& name)
{

}

void Scene::addGameObject(std::shared_ptr<GameObject> game_object)
{
	root->addChild(game_object, root);
}

void Scene::addComponentToGameObject(std::shared_ptr<Component>& component, std::shared_ptr<GameObject> game_object)
{
	game_object->addComponent(component, game_object);
}

std::shared_ptr<Camera> Scene::getCamera() const
{
	return camera;
}

void Scene::drawGUI()
{
	//static float f = 0.0f;
	//static int counter = 0;
	//static glm::vec3 clear_color;

	//ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

	//ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)

	//ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
	//ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

	//if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
	//	counter++;
	//ImGui::SameLine();
	//ImGui::Text("counter = %d", counter);

	//ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	//ImGui::End();
	bool show = true;
	ImGui::ShowDemoWindow(&show);
}
