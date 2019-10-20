#ifndef COMPONENT_H
#define COMPONENT_H

#include <GUI/ImGui/imgui.h>

#include "EngineSystems/SceneManager.h"
#include "ISerializable.h"
#include "Scene.h"

namespace spark {

class GameObject;
class Component : public std::enable_shared_from_this<Component> ,public ISerializable
{
public:
	std::string name = "Component";
	
	Component() = default;
	Component(std::string& componentName);
	virtual ~Component();
	virtual void update() = 0;
	virtual void fixedUpdate() = 0;
	
	void drawComponentGUI();
	void beginDrawingWindow();
	virtual void drawGUI() = 0;
	void endDrawingWindow();

	std::shared_ptr<GameObject> getGameObject() const;
	bool getActive() const;
	void setGameObject(std::shared_ptr<GameObject>& game_object);
	virtual void setActive(bool active_);

	template <class T>
	void removeComponent()
	{
		auto remove = [this]()
		{
			getGameObject()->removeComponent<T>(shared_from_base<T>());
		};
		SceneManager::getInstance()->getCurrentScene()->toRemove.push_back(remove);
	}

	template <class T>
	void removeComponentGUI()
	{
		if (ImGui::Button("Delete"))
		{
			removeComponent<T>();
		}
	};

protected:
	bool active = true;

	template <typename Derived>
	std::shared_ptr<Derived> shared_from_base()
	{
		return std::static_pointer_cast<Derived>(shared_from_this());
	}

private:
	std::weak_ptr<GameObject> gameObject;
	
};

}
#endif