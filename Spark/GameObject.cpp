#include <GameObject.h>
#include <algorithm>
#include <iostream>
#include <GUI/ImGui/imgui.h>

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

void GameObject::addChild(const std::shared_ptr<GameObject>& newChild, const std::shared_ptr<GameObject>& parent)
{
	newChild->parent = parent;
	children.push_back(newChild);
}

void GameObject::addComponent(std::shared_ptr<Component>& component, std::shared_ptr<GameObject>& gameObject)
{
	component->setGameObject(gameObject);
	components.push_back(component);
}

bool GameObject::removeComponent(std::string&& name)
{
	const auto component_it = std::find_if(std::begin(components), std::end(components), [&name] (const std::shared_ptr<Component>& component)
	{
		return component->name == name;
	});
	if(component_it != components.end())
	{
		components.erase(component_it);
		return true;
	}
	return false;
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

bool GameObject::removeChild(std::shared_ptr<GameObject>& child)
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
	transform.local.drawGUI();
	for (auto& component : components)
		component->drawGUI();
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
