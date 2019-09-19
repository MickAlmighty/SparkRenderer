#include <Scene.h>
#include <list>
#include <GUI/ImGui/imgui.h>
#include "Spark.h"
#include "JsonSerializer.h"
#include "HID.h"
#include "GUI/ImGuizmo.h"
#include <glm/gtc/type_ptr.hpp>

Scene::Scene(std::string&& sceneName) : name(sceneName)
{

}

Scene::~Scene()
{
#ifdef DEBUG
	std::cout << "Scene destroyed!" << std::endl;
#endif
}

void Scene::update()
{
	removeObjectsFromScene();
	
	camera->ProcessKeyboard();
	camera->ProcessMouseMovement(HID::mouse.direction.x, -HID::mouse.direction.y);
	root->update();
}

void Scene::fixedUpdate()
{
	root->fixedUpdate();
}

void Scene::removeObjectsFromScene()
{
	std::for_each(std::begin(toRemove), std::end(toRemove), [](std::function<void()>& f)
	{
		f();
	});
	toRemove.clear();
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

Json::Value Scene::serialize() const
{
	Json::Value serialize;
	serialize["name"] = name;
	Json::Value serializeSceneGraph;
	serialize["sceneGraph"] = JsonSerializer::serialize(root);
	return serialize;
}

void Scene::deserialize(Json::Value& deserializationRoot)
{
	for (auto& member : deserializationRoot.getMemberNames())
	{
		std::cout<< member << std::endl;
	}
	name = deserializationRoot.get("name", "Scene").asString();
	root = std::static_pointer_cast<GameObject>(JsonSerializer::deserialize(deserializationRoot["sceneGraph"]));
}

void Scene::drawGUI()
{
	drawSceneGraph();
	static bool opened = false;
	ImGuiIO& io = ImGui::GetIO();
	
	ImGui::SetNextWindowPos({ io.DisplaySize.x - 5, 25 }, ImGuiCond_Always, {1, 0} );
	ImGui::SetNextWindowSizeConstraints(ImVec2(250, 120), ImVec2(FLT_MAX, io.DisplaySize.y - 50)); // Width = 250, Height > 100
	if (ImGui::Begin("GameObject", &opened, {0, 0}, -1, ImGuiWindowFlags_AlwaysAutoResize))
	{
		auto gameObject_ptr = gameObjectToPreview.lock();
		if (gameObject_ptr != nullptr)
		{
			gameObject_ptr->drawGUI();

			static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::ROTATE);
			static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::WORLD);
			if (ImGui::IsKeyPressed(90))
				mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
			if (ImGui::IsKeyPressed(69))
				mCurrentGizmoOperation = ImGuizmo::ROTATE;
			if (ImGui::IsKeyPressed(82)) // r Key
				mCurrentGizmoOperation = ImGuizmo::SCALE;
			if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
				mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
			ImGui::SameLine();
			if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
				mCurrentGizmoOperation = ImGuizmo::ROTATE;
			ImGui::SameLine();
			if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
				mCurrentGizmoOperation = ImGuizmo::SCALE;

			glm::vec3 pos;// = gameObject_ptr->transform.world.getPosition();
			glm::vec3 scale;// = gameObject_ptr->transform.world.getScale();
			glm::vec3 rotation;// = gameObject_ptr->transform.world.getRotationDegrees();
			ImGuizmo::DecomposeMatrixToComponents(&gameObject_ptr->transform.local.getMatrix()[0][0], &pos.x, &rotation.x, &scale.x);

			ImGui::InputFloat3("Tr", glm::value_ptr(pos), 3);
			ImGui::InputFloat3("Sc", glm::value_ptr(scale), 3);
			ImGui::InputFloat3("Rt", glm::value_ptr(rotation), 3);

			glm::mat4 mat(1);
			ImGuizmo::RecomposeMatrixFromComponents(&pos.x, &rotation.x, &scale.x, glm::value_ptr(mat));


			if (mCurrentGizmoOperation != ImGuizmo::SCALE)
			{
				if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
					mCurrentGizmoMode = ImGuizmo::LOCAL;
				ImGui::SameLine();
				if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
					mCurrentGizmoMode = ImGuizmo::WORLD;
			}
		
			ImGuiIO& io = ImGui::GetIO();
			ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
			ImGuizmo::Manipulate(glm::value_ptr(camera->GetViewMatrix()), glm::value_ptr(camera->getProjectionMatrix()), mCurrentGizmoOperation, mCurrentGizmoMode, &mat[0][0]);
			
		/*	glm::mat4 worldToLocal = glm::inverse(mat);*/
			ImGuizmo::DecomposeMatrixToComponents(&mat[0][0], &pos.x, &rotation.x, &scale.x);
			
			gameObject_ptr->transform.local.setPosition(pos);
			gameObject_ptr->transform.local.setScale(scale);
			gameObject_ptr->transform.local.setRotationDegrees(rotation);
		}
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
		drawTreeNode(root, true);
	}
	ImGui::End();

	std::for_each(std::begin(toRemove), std::end(toRemove), [](std::function<void()>& f)
	{
		f();
	});
	toRemove.clear();
}

void Scene::drawTreeNode(std::shared_ptr<GameObject> node, bool isRootNode)
{
	ImGui::PushID(node.get());
	bool opened = node == gameObjectToPreview.lock();
	ImGui::Selectable(node->name.c_str(), opened, 0, { node->name.length() * 7.1f, 0.0f });
	ImGui::OpenPopupOnItemClick("GraphNode_operations", 1);

	if (ImGui::BeginPopup("GraphNode_operations"))
	{
		ImGui::Text("Popup Example");

		if (ImGui::Button("Add child"))
		{
			node->addChild(std::make_shared<GameObject>(), node);
			ImGui::CloseCurrentPopup();
		}

		if(!isRootNode)
		{
			if (ImGui::Button("Delete"))
			{
				const std::function<void()> remove = [node]()
				{
					node->getParent()->removeChild(node);
				};
				toRemove.push_back(remove);
				ImGui::CloseCurrentPopup();
			}
		}
		ImGui::EndPopup();
	}

	if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0) && !isRootNode)
	{
		gameObjectToPreview = node;
		std::cout << "GameObject: " + node->name + " clicked!" << std::endl;
	}

	if (!node->children.empty())
	{
		ImGui::SameLine();
		if (ImGui::TreeNode("Children"))
		{
			for (auto gameObject : node->children)
			{
				drawTreeNode(gameObject, false);
			}

			ImGui::TreePop();
		}
	}
	ImGui::PopID();
}
