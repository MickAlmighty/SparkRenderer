#pragma once
#include <list>
#include <memory>
#include <Component.h>
#include <Scene.h>
#include <Structs.h>
#include <ISerializable.h>

class Component;
class GameObject : public std::enable_shared_from_this<GameObject>, public ISerializable
{
private:
	friend class Scene;
	std::string name = "GameObject";
	std::weak_ptr<GameObject> parent;
	std::list<std::shared_ptr<GameObject>> children;
	std::list<std::shared_ptr<Component>> components;
	std::shared_ptr<GameObject> get_ptr();
	void update();
	void fixedUpdate();
	void drawGizmos();
public:
	Transform transform;
	std::shared_ptr<GameObject> getParent() const;
	void setParent(const std::shared_ptr<GameObject> newParent);
	void addChild(const std::shared_ptr<GameObject>& newChild, const std::shared_ptr<GameObject>& parent);
	void addComponent(std::shared_ptr<Component> component, std::shared_ptr<GameObject> gameObject);
	bool removeChild(std::string&& gameObjectName);
	bool removeChild(std::shared_ptr<GameObject> child);
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

	template <class T>
	std::shared_ptr<T> getComponent()
	{
		auto component_it = std::find_if(std::begin(components), std::end(components), [](std::shared_ptr<Component> component)
		{
			T* comp_ptr = dynamic_cast<T*>(component.get());
			return comp_ptr != nullptr;
		});
		if(component_it != components.end())
		{
			return *component_it;
		}
		return nullptr;
	}

	template <class T>
	bool removeComponent(std::string& name)
	{
		auto component_it = std::find_if(std::begin(components), std::end(components), [&name](std::shared_ptr<Component> component)
		{
			if(dynamic_cast<T*>(component.get()))
			{
				return component->name == name;
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

	SerializableType getSerializableType() override;
	Json::Value serialize() override;
	void deserialize(Json::Value& root) override;


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

