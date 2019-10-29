#ifndef COMPONENT_H
#define COMPONENT_H

#include "EngineSystems/SceneManager.h"
#include "ISerializable.h"
#include "Scene.h"

#include <rttr/registration_friend>
#include <rttr/registration>
#include <GUI/ImGui/imgui.h>

namespace spark {

	class GameObject;
	class Component abstract {
	public:
		Component() = default;
		Component(const std::string& componentName);
		virtual ~Component();
		Component(Component&) = delete;
		Component(Component&&) = delete;
		Component& operator=(const Component&) = delete;
		Component& operator=(Component&&) = delete;

		virtual void update() = 0;
		virtual void fixedUpdate() = 0;
		void drawComponentGUI();
		void beginDrawingWindow();
		virtual void drawGUI() = 0;
		void endDrawingWindow();
		std::shared_ptr<GameObject> getGameObject() const;
		bool getActive() const;
		void setGameObject(std::shared_ptr<GameObject>& game_object);
		void setActive(bool active_);
		template <class T>
		void removeComponentGUI() {
			if (ImGui::Button("Delete")) {
				auto remove = [this]() {
					getGameObject()->removeComponent<T>(name);
				};
				SceneManager::getInstance()->getCurrentScene()->toRemove.push_back(remove);
			}
		};

		std::string name{ "Component" };
	private:
		bool active = true;
		std::weak_ptr<GameObject> gameObject;
		RTTR_REGISTRATION_FRIEND;
		RTTR_ENABLE()
	};
}

RTTR_REGISTRATION{
	rttr::registration::class_<spark::Component>("Component")
	.property("active", &spark::Component::active)
	.property("name", &spark::Component::name);
}
#endif