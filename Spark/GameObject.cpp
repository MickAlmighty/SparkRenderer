#include "GameObject.h"

#include <algorithm>
#include <iostream>

#include <GUI/ImGui/imgui.h>
#include <GUI/ImGuizmo.h>
#include "glm/gtc/type_ptr.hpp"

#include "Camera.h"
#include "Component.h"
#include "JsonSerializer.h"
#include "Scene.h"
#include "GUI/SparkGui.h"
#include "Logging.h"

namespace spark {

void GameObject::update()
{
	if (parent.lock() == nullptr)
	{
		transform.world.setMatrix(transform.local.getMatrix());
	}
	else
	{
		transform.world.setMatrix(parent.lock()->transform.world.getMatrix() * transform.local.getMatrix());
	}

	for (auto& component : components)
	{
		if(component->getActive())
		{
			component->update();
		}
	}

	for (auto& child : children)
	{
		child->update();
	}
}

void GameObject::fixedUpdate()
{
	/*if(parent.lock() == nullptr)
	{
		transform.world.setMatrix(transform.local.getMatrix());
	}
	else
	{
		transform.world.setMatrix(parent.lock()->transform.world.getMatrix() * transform.local.getMatrix());
	}*/

	for (auto& component : components)
	{
		component->fixedUpdate();
	}
}

std::shared_ptr<GameObject> GameObject::getParent() const
{
	return parent.lock();
}

std::shared_ptr<Scene> GameObject::getScene() const {
	return scene.lock();
}

	void GameObject::setParent(const std::shared_ptr<GameObject> newParent)
{
	if (!parent.expired())
	{
		parent.lock()->removeChild(shared_from_this());
	}

	parent = newParent;
}

void GameObject::setScene(const std::shared_ptr<Scene> newScene) {
	this->scene = newScene;
}

	void GameObject::addChild(const std::shared_ptr<GameObject>& newChild, const std::shared_ptr<GameObject>& parent)
{
	newChild->setParent(parent);
	newChild->setScene(getScene());
	children.push_back(newChild);
}

void GameObject::addComponent(std::shared_ptr<Component> component)
{
	component->setGameObject(shared_from_this());
	components.push_back(component);
}

bool GameObject::removeChild(std::string&& gameObjectName)
{
	const auto gameObject_it = std::find_if(std::begin(children), std::end(children), [&gameObjectName](const std::shared_ptr<GameObject>& gameObject)
	{
		return gameObject->name == gameObjectName;
	});
	if (gameObject_it != children.end())
	{
		children.erase(gameObject_it);
		return true;
	}
	return false;
}

bool GameObject::removeChild(std::shared_ptr<GameObject> child)
{
	const auto gameObject_it = std::find_if(std::begin(children), std::end(children), [&child](const std::shared_ptr<GameObject>& gameObject)
	{
		return gameObject == child;
	});
	if (gameObject_it != children.end())
	{
		children.erase(gameObject_it);
		return true;
	}
	return false;
}

void GameObject::drawGUI()
{
	ImGui::Text(name.c_str());
	static char nameInput[64] = "";
	ImGui::InputTextWithHint("", name.c_str(), nameInput, 64, ImGuiInputTextFlags_CharsNoBlank);
	ImGui::SameLine();
	if (ImGui::Button("Change Name") && nameInput[0] != '\0')
	{
		name = nameInput;
		for (int i = 0; i < 64; i++)
		{
			nameInput[i] = '\0';
		}
	}
	transform.local.drawGUI();
	drawGizmos();
	for (auto& component : components)
		component->drawComponentGUI();

	ImGui::NewLine();
	const std::shared_ptr<Component> componentToAdd = SparkGui::addComponent();
	if (componentToAdd != nullptr)
	{
		addComponent(componentToAdd);
	}
}

std::string GameObject::getName() const {
	return name;
}

	void GameObject::drawGizmos()
{
	static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::ROTATE);
	static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::LOCAL);
	if (ImGui::IsKeyPressed(GLFW_KEY_T))
		mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
	if (ImGui::IsKeyPressed(GLFW_KEY_R))
		mCurrentGizmoOperation = ImGuizmo::ROTATE;
	if (ImGui::IsKeyPressed(GLFW_KEY_Y))
	{
		mCurrentGizmoOperation = ImGuizmo::SCALE;
		mCurrentGizmoMode = ImGuizmo::LOCAL;
	}

	if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
		mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
	ImGui::SameLine();
	if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
		mCurrentGizmoOperation = ImGuizmo::ROTATE;
	ImGui::SameLine();
	if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
		mCurrentGizmoOperation = ImGuizmo::SCALE;

	glm::mat4 worldMatrix = getParent()->transform.world.getMatrix() * transform.local.getMatrix();//transform.world.getMatrix();

	if (mCurrentGizmoOperation != ImGuizmo::SCALE)
	{
		if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
			mCurrentGizmoMode = ImGuizmo::LOCAL;
		ImGui::SameLine();
		if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
			mCurrentGizmoMode = ImGuizmo::WORLD;
	}

	ImGuiIO& io = ImGui::GetIO();
	auto camera = getScene()->getCamera();
	ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
	ImGuizmo::Manipulate(glm::value_ptr(camera->getViewMatrix()), glm::value_ptr(camera->getProjectionMatrix()), mCurrentGizmoOperation, mCurrentGizmoMode, &worldMatrix[0][0]);

	glm::mat4 localMatrix = glm::inverse(getParent()->transform.world.getMatrix()) * worldMatrix; //getting new localTransform by multiplying by inverse of parent world transform

	glm::vec3 pos{}, scale{}, rotation{};
	ImGuizmo::DecomposeMatrixToComponents(&localMatrix[0][0], &pos.x, &rotation.x, &scale.x);


	if (glm::length(pos - transform.local.getPosition()) > 0.1f)
	{
		transform.local.setPosition(pos);
	}
	if (glm::length(scale - transform.local.getScale()) > 0.1f)
	{
		transform.local.setScale(scale);
	}
	if (glm::length(rotation - transform.local.getRotationDegrees()) > 0.1f)
	{
		transform.local.setRotationDegrees(rotation);
	}
}

    GameObject::GameObject(std::string&& name) : name(std::move(name)) {}

GameObject::~GameObject()
{
    SPARK_TRACE("GameObject '{}' destroyed!", name);
}

}

RTTR_REGISTRATION{
    rttr::registration::class_<spark::GameObject>("GameObject")
    .constructor()(rttr::policy::ctor::as_std_shared_ptr)
    .property("transform", &spark::GameObject::transform)
    .property("name", &spark::GameObject::name)
    .property("scene", &spark::GameObject::getScene, &spark::GameObject::setScene, rttr::registration::public_access)
    .property("parent", &spark::GameObject::getParent, &spark::GameObject::setParent, rttr::registration::public_access)
    .property("children", &spark::GameObject::children)
    .property("components", &spark::GameObject::components);
}