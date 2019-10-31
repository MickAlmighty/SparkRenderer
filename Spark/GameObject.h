#ifndef GAME_OBJECT_H
#define GAME_OBJECT_H

#include <list>
#include <memory>

#include "ISerializable.h"
#include "Structs.h"
#include "Component.h"

namespace spark {


class GameObject final : public std::enable_shared_from_this<GameObject>
{
public:
	Transform transform;

	GameObject(std::string&& _name = "GameObject");
	~GameObject();
	GameObject(GameObject&) = delete;
	GameObject(GameObject&&) = delete;
	GameObject& operator=(const GameObject&) = delete;
	GameObject& operator=(GameObject&&) = delete;
	
	std::shared_ptr<GameObject> getParent() const;
	std::shared_ptr<Scene> getScene() const;
	void setParent(const std::shared_ptr<GameObject> newParent);
	void setScene(const std::shared_ptr<Scene> newScene);
	void addChild(const std::shared_ptr<GameObject>& newChild, const std::shared_ptr<GameObject>& parent);
	void addComponent(std::shared_ptr<Component> component);
	bool removeChild(std::string&& gameObjectName);
	bool removeChild(std::shared_ptr<GameObject> child);
	void drawGUI();
	std::string getName() const;

	template <class T>
	bool removeFirstComponentOfType()
	{
		auto it = std::begin(components);
		for (auto& component : components)
		{
			T* component_ptr = dynamic_cast<T*>(component.get());
			if (component_ptr != nullptr)
			{
				components.erase(it);
				return true;
			}
			++it;
		}
		return false;
	}

	template <class T>
	std::shared_ptr<T> getComponent()
	{
		auto component_it = std::find_if(std::begin(components), std::end(components), [](std::shared_ptr<Component> component)
		{
			T* comp_ptr = dynamic_cast<T*>(component.get());
			return comp_ptr != nullptr;
		});
		if (component_it != components.end())
		{
			return std::dynamic_pointer_cast<T>(*component_it);
		}
		return nullptr;
	}

	template <class T>
	bool removeComponent(std::string& name)
	{
		auto component_it = std::find_if(std::begin(components), std::end(components), [&name](const std::shared_ptr<Component>& component)
		{
			if (dynamic_cast<T*>(component.get()))
			{
				return component->getName() == name;
			}
			return false;
		});
		if (component_it != components.end())
		{
			components.erase(component_it);
			return true;
		}
		return false;
	}

	template <class T>
	bool removeComponent(const std::shared_ptr<T>& c)
	{
		auto component_it = std::find_if(std::begin(components), std::end(components), [&c](const std::shared_ptr<Component>& component)
		{
			if (dynamic_cast<T*>(component.get()))
			{
				return component.get() == c.get();
			}
			return false;
		});
		if (component_it != components.end())
		{
			components.erase(component_it);
			return true;
		}
		return false;
	}

	/*template <class T>
	bool removeComponentsOfType()
	{
		for (auto& component : components)
		{
			T* component_ptr = dynamic_cast<T*>(component.get());
			if (component_ptr != nullptr)
			{
				components.remove(component);
				return true;
			}
		}
		return false;
	}*/

private:
	friend class Scene;
	std::string name = "GameObject";
	std::weak_ptr<Scene> scene;
	std::weak_ptr<GameObject> parent;
	std::list<std::shared_ptr<GameObject>> children;
	std::list<std::shared_ptr<Component>> components;
	void update();
	void fixedUpdate();
	void drawGizmos();

};

}
#endif