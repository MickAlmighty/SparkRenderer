#include <Scene.h>
#include <list>
#include <GUI/ImGui/imgui.h>
#include "Spark.h"

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
	/*for(GameObject gameObject : root)
	ImGui::TreePush()*/
	drawSceneGraph();
	static bool opened = false;
	ImGuiIO& io = ImGui::GetIO();
	
	ImGui::SetNextWindowPos({ io.DisplaySize.x - 5, 25 }, ImGuiCond_Always, {1, 0} );
	ImGui::SetNextWindowSizeConstraints(ImVec2(250, 100), ImVec2(FLT_MAX, FLT_MAX)); // Width > 250, Height > 100
	if (ImGui::Begin("GameObject", &opened, {0, 0}, -1, ImGuiWindowFlags_AlwaysAutoResize))
	{
		auto gameObject_ptr = gameObjectToPreview.lock();
		if (gameObject_ptr != nullptr)
			gameObject_ptr->drawGUI();
	}
	ImGui::End();
}

void Scene::drawSceneGraph()
{
	ImGuiIO& io = ImGui::GetIO();
	ImGui::SetNextWindowPos({ 5, 25 }, ImGuiCond_Always, { 0, 0 });
	if (ImGui::Begin("Scene", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_HorizontalScrollbar))
	{
		ImGui::Text(name.c_str());
		ImGui::Separator();
		drawTreeNode(root);
	}
	ImGui::End();
}

void Scene::drawTreeNode(std::shared_ptr<GameObject>& node)
{
	bool opened = node == gameObjectToPreview.lock();
	ImGui::Selectable(node->name.c_str(), opened, 0, { node->name.length() * 7.1f, 0.0f });
	ImGui::OpenPopupOnItemClick("GraphNode_operations", 1);

	if (ImGui::BeginPopup("GraphNode_operations"))
	{
		ImGui::Text("Popup Example");
		ImGui::EndPopup();
	}

	if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
	{
		gameObjectToPreview = node;
		std::cout << "GameObject: " + node->name + " clicked!" << std::endl;
	}

	if (node->children.empty())
		return;

	ImGui::SameLine();
	if (ImGui::TreeNode("Children"))
	{
		ImGui::PushID(this);
		for (auto& gameObject : node->children)
		{
			drawTreeNode(gameObject);
		}

		ImGui::TreePop();
		ImGui::PopID();
	}
}
