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
	inline virtual void update() = 0;
	virtual void fixedUpdate() = 0;
	
	void drawComponentGUI();
	inline void beginDrawingWindow();
	inline virtual void drawGUI() = 0;
	inline void endDrawingWindow();

	inline std::shared_ptr<GameObject> getGameObject() const
	{
		return gameObject.lock();
	}
	
	inline bool getActive() const
	{
		return active;
	}

	inline void setGameObject(std::shared_ptr<GameObject>& game_object)
	{
		gameObject = game_object;
	}

	inline virtual 	void setActive(bool active_)
	{
		active = active_;
	}

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