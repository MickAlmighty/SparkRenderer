#include "Component.h"

#include <iostream>

namespace spark {
	Component::Component(const std::string& componentName) {
		name = componentName;
	}

	Component::~Component() {
#ifdef DEBUG
		std::cout << "Component: " + name << " destroyed!" << std::endl;
#endif
	}

	void Component::drawComponentGUI() {
		beginDrawingWindow();
		if (active) {
			drawGUI();
		}
		endDrawingWindow();
	}

	void Component::beginDrawingWindow() {
		ImGui::PushID(this);
		//ImGui::Separator();
		ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
		//ImGui::SetNextWindowSizeConstraints(ImVec2(0, 0), ImVec2(FLT_MAX, FLT_MAX));
		ImGui::BeginChild(name.c_str(), { 320, 180 }, true, ImGuiWindowFlags_MenuBar);
		if (ImGui::BeginMenuBar()) {
			ImGui::Text(name.c_str());
			bool active = getActive();
			ImGui::SameLine();
			ImGui::Checkbox("Enabled", &active);
			if (active != getActive()) {
				setActive(active);
			}
			ImGui::EndMenuBar();
		}
	}

	void Component::endDrawingWindow() {
		ImGui::EndChild();
		ImGui::PopStyleVar();
		//ImGui::Separator();
		ImGui::PopID();
	}

	std::shared_ptr<GameObject> Component::getGameObject() const {
		return gameObject.lock();
	}

	bool Component::getActive() const {
		return active;
	}

	void Component::setGameObject(std::shared_ptr<GameObject>& game_object) {
		gameObject = game_object;
	}

	void Component::setActive(bool active_) {
		active = active_;
	}
}