#include "GameObject.h"
#include <algorithm>

void GameObject::update()
{
}

void GameObject::fixedUpdate()
{
	if(parent == nullptr)
	{
		world = local;
	}
	else
	{
		//world = parent->world.getMatrix() * local.getMatrix();
	}
}

void GameObject::addChild(std::shared_ptr<GameObject>&& newChild, std::shared_ptr<GameObject>&& parent)
{
	newChild->parent = parent;
	children.push_back(newChild);
}

void GameObject::addComponent(std::shared_ptr<Component> component)
{
	components.push_back(component);
}

bool GameObject::removeComponent(std::string&& name)
{
	const auto component_it = std::find_if(std::begin(components), std::end(components), [name] (const std::shared_ptr<Component>& component)
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
	const auto gameObject_it = std::find_if(std::begin(children), std::end(children), [gameObjectName] (const std::shared_ptr<GameObject>& gameObject)
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

GameObject::GameObject(std::string&& _name) : name(_name)
{

}

GameObject::~GameObject()
{
	parent = nullptr;
	children.clear();
}
