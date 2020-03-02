#include "Component.h"
#include "GameObject.h"

#include <iostream>
#include "Logging.h"

namespace spark
{
Component::Component(std::string&& name) : name(std::move(name)) {}

Component::~Component()
{
    SPARK_TRACE("Component '{}' destroyed!", name);
}

void Component::drawComponentGUI()
{
    beginDrawingWindow();
    if(active)
    {
        drawGUI();
    }
    endDrawingWindow();
}

void Component::beginDrawingWindow()
{
    ImGui::PushID(this);
    // ImGui::Separator();
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
    // ImGui::SetNextWindowSizeConstraints(ImVec2(0, 0), ImVec2(FLT_MAX, FLT_MAX));
    ImGui::BeginChild(name.c_str(), {320, 180}, true, ImGuiWindowFlags_MenuBar);
    if(ImGui::BeginMenuBar())
    {
        ImGui::Text(name.c_str());
        bool active = getActive();
        ImGui::SameLine();
        ImGui::Checkbox("Enabled", &active);
        if(active != getActive())
        {
            setActive(active);
        }
        ImGui::EndMenuBar();
    }
}

void Component::endDrawingWindow()
{
    ImGui::EndChild();
    ImGui::PopStyleVar();
    // ImGui::Separator();
    ImGui::PopID();
}

std::shared_ptr<GameObject> Component::getGameObject() const
{
    return gameObject.lock();
}

std::string Component::getName() const
{
    return name;
}

bool Component::getActive() const
{
    return active;
}

void Component::setGameObject(const std::shared_ptr<GameObject> game_object)
{
    gameObject = game_object;
}

void Component::setActive(bool active_)
{
    active = active_;
}

std::shared_ptr<Component> Component::getComponentPtr()
{
    return std::static_pointer_cast<Component>(shared_from_this());
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::Component>("Component")
        .property("active", &spark::Component::active)
        .property("name", &spark::Component::name)
        .method("getComponentPtr", &spark::Component::getComponentPtr)
        .property("gameObject", &spark::Component::getGameObject, &spark::Component::setGameObject, rttr::registration::public_access);
}