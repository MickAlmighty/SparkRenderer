#ifndef COMPONENT_H
#define COMPONENT_H

#include <GUI/ImGui/imgui.h>

#include "EngineSystems/SceneManager.h"
#include "ISerializable.h"
#include "Scene.h"

namespace spark {

class GameObject;
class Component : public ISerializable
{
public:
	std::string name = "Component";
	
	Component() = default;
	Component(std::string& componentName);
	virtual ~Component();

	std::shared_ptr<GameObject> getGameObject() const { return gameObject.lock(); }
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

private:
	std::weak_ptr<GameObject> gameObject;
};

}
#endif