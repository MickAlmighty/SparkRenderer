#pragma once
#include <list>
#include <memory>
#include <Component.h>
#include <Scene.h>
#include <Structs.h>

class Component;
class GameObject
{
private:
	friend class Scene;
	std::string name = "GameObject";
	std::weak_ptr<GameObject> parent;
	std::list<std::shared_ptr<GameObject>> children;
	std::list<std::shared_ptr<Component>> components;
	void update();
	void fixedUpdate();
public:
	Transform transform;
	std::shared_ptr<GameObject> getParent() const;
	void addChild(const std::shared_ptr<GameObject>& newChild, const std::shared_ptr<GameObject>& parent);
	void addComponent(std::shared_ptr<Component>& component, std::shared_ptr<GameObject>& gameObject);
	bool removeComponent(std::string&& name);
	bool removeChild(std::string&& gameObjectName);
	bool removeChild(std::shared_ptr<GameObject>& child);
	void drawGUI();
	GameObject(std::string&& _name = "GameObject");
	~GameObject();

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
};

