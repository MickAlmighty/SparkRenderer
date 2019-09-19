#pragma once
#include <iostream>
#include <GameObject.h>
#include "GUI/ImGui/imgui.h"
#include <ISerializable.h>

class GameObject;
class Component : public ISerializable
{
private:
	std::weak_ptr<GameObject> gameObject;
public:
	std::string name = "Component";
	Component() = default;
	Component(std::string& componentName);
	std::shared_ptr<GameObject> getGameObject() const { return gameObject.lock(); }
	virtual ~Component() = default;
	virtual void update() = 0;
	virtual void fixedUpdate() = 0;
	virtual void setGameObject(std::shared_ptr<GameObject>& game_object) { gameObject = game_object; };
	virtual void drawGUI() = 0;
	template <class T>
	void removeComponentGUI()
	{
		if (ImGui::Button("Delete"))
		{
			auto remove = [this]()
			{
				getGameObject()->removeComponent<T>(name);
			};
			SceneManager::getInstance()->getCurrentScene()->toRemove.push_back(remove);
		}
	};
};

