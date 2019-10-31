#include "Scene.h"

#include <iostream>
#include <list>
#include <utility>

#include <GUI/ImGui/imgui.h>

#include <Camera.h>
#include <HID.h>
#include <GameObject.h>
#include <JsonSerializer.h>

namespace spark {

Scene::Scene(std::string sceneName) : name(std::move(sceneName))
{
    root = std::make_shared<GameObject>("Root");
	camera = std::make_shared<Camera>(glm::vec3(0, 0, 5));
	lightManager = std::make_unique<LightManager>();
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
	if (cameraMovement)
	{
		camera->processKeyboard();
		camera->processMouseMovement(HID::mouse.direction.x, -HID::mouse.direction.y);
	}
	camera->update();
	root->update();
	lightManager->updateLightBuffers();
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

std::shared_ptr<Camera> Scene::getCamera() const
{
	return camera;
}

std::shared_ptr<GameObject> Scene::getRoot() const {
    return root;
}

    void Scene::drawGUI()
{
	drawSceneGraph();
	static bool opened = false;
	ImGuiIO& io = ImGui::GetIO();

	ImGui::SetNextWindowPos({ io.DisplaySize.x - 5, 25 }, ImGuiCond_Always, { 1, 0 });
	ImGui::SetNextWindowSizeConstraints(ImVec2(350, 20), ImVec2(350, io.DisplaySize.y - 50));
	if (ImGui::Begin("GameObject", &opened, { 0, 0 }, -1, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_AlwaysHorizontalScrollbar | ImGuiWindowFlags_NoCollapse))
	{
		auto gameObject_ptr = gameObjectToPreview.lock();
		if (gameObject_ptr != nullptr)
		{
			camera->setCameraTarget(gameObject_ptr->transform.world.getPosition());
			if (ImGui::Button("Close Preview"))
			{
				gameObjectToPreview.reset();
			}
			else
			{
				gameObject_ptr->drawGUI();
			}
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

	if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None) && !isRootNode)
	{
		ImGui::SetDragDropPayload("OBJECT_DRAG_AND_DROP", &node, sizeof(std::shared_ptr<GameObject>));        // Set payload to carry the index of our item (could be anything)
		ImGui::Text("Getting reference to %s", node->name.c_str());
		ImGui::EndDragDropSource();
	}

	ImGui::OpenPopupOnItemClick("GraphNode_operations", 1);

	if (ImGui::BeginPopup("GraphNode_operations"))
	{
		ImGui::Text("Popup Example");

		if (ImGui::Button("Add child"))
		{
			node->addChild(std::make_shared<GameObject>(), node);
			ImGui::CloseCurrentPopup();
		}

		if (!isRootNode)
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
			for (const auto& gameObject : node->children)
			{
				drawTreeNode(gameObject, false);
			}

			ImGui::TreePop();
		}
	}
	ImGui::PopID();
}

}