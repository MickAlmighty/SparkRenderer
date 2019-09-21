#include <GameObject.h>
#include <algorithm>
#include <iostream>
#include <GUI/ImGui/imgui.h>
#include <GUI/SparkGui.h>
#include "JsonSerializer.h"
#include "GUI/ImGuizmo.h"
#include <Component.h>
#include <Scene.h>
#include <Camera.h>
#include <glm/gtc/type_ptr.hpp>

std::shared_ptr<GameObject> GameObject::get_ptr()
{
	return shared_from_this();
}

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

	for(auto& component: components)
	{
		component->update();
	}

	for(auto& child: children)
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

void GameObject::setParent(const std::shared_ptr<GameObject> newParent)
{
	if(!parent.expired())
	{
		parent.lock()->removeChild(shared_from_this());
	}

	parent = newParent;
}

void GameObject::addChild(const std::shared_ptr<GameObject>& newChild, const std::shared_ptr<GameObject>& parent)
{
	newChild->setParent(parent);
	children.push_back(newChild);
}

void GameObject::addComponent(std::shared_ptr<Component> component, std::shared_ptr<GameObject> gameObject)
{
	component->setGameObject(gameObject);
	components.push_back(component);
}

bool GameObject::removeChild(std::string&& gameObjectName)
{
	const auto gameObject_it = std::find_if(std::begin(children), std::end(children), [&gameObjectName] (const std::shared_ptr<GameObject>& gameObject)
	{
		return gameObject->name == gameObjectName;
	});
	if(gameObject_it != children.end())
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
	if(ImGui::Button("Change Name") && nameInput[0] != '\0')
	{
		name = nameInput;
		for(int i = 0; i < 64; i++)
		{
			nameInput[i] = '\0';
		}
	}
	transform.local.drawGUI();
	drawGizmos();
	for (auto& component : components)
		component->drawGUI();

	const std::shared_ptr<Component> componentToAdd = SparkGui::addComponent();
	if(componentToAdd != nullptr)
	{
		addComponent(componentToAdd, get_ptr());
	}
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
	auto camera = SceneManager::getInstance()->getCurrentScene()->getCamera();
	ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
	ImGuizmo::Manipulate(glm::value_ptr(camera->GetViewMatrix()), glm::value_ptr(camera->getProjectionMatrix()), mCurrentGizmoOperation, mCurrentGizmoMode, &worldMatrix[0][0]);

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


GameObject::GameObject(std::string&& _name) : name(_name)
{
	
}

GameObject::~GameObject()
{
#ifdef DEBUG
	std::cout << "GameObject: " << name.c_str() << " destroyed!" << std::endl;
#endif
}

SerializableType GameObject::getSerializableType()
{
	return SerializableType::SGameObject;
}

Json::Value GameObject::serialize()
{
	Json::Value root;
	root["name"] = name;
	root["localTransform"] = transform.local.serialize();
	root["worldTransform"] = transform.world.serialize();
	
	unsigned int j = 0;
	for(const auto& component : components)
	{
		root["components"][j] = JsonSerializer::serialize(component);
		++j;
	}

	unsigned int i = 0;
	for(const auto& child : children)
	{
		root["children"][i] = JsonSerializer::serialize(child);
		++i;
	}
	return root;
}

void GameObject::deserialize(Json::Value& root)
{
	name = root.get("name", "GameObject").asString();
	transform.local.deserialize(root["localTransform"]);
	transform.world.deserialize(root["worldTransform"]);
	for(unsigned int i = 0; i < root["children"].size(); ++i)
	{
		auto child = std::static_pointer_cast<GameObject>(JsonSerializer::deserialize(root["children"][i]));
		addChild(child, shared_from_this());
	}

	for (unsigned int i = 0; i < root["components"].size(); ++i)
	{
		const auto component = std::static_pointer_cast<Component>(JsonSerializer::deserialize(root["components"][i]));
		addComponent(component, shared_from_this());
	}
}
