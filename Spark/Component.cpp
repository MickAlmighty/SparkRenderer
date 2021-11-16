#include "Component.h"

#include "GUI/ImGui/imgui_custom_widgets.h"
#include "GameObject.h"
#include "Logging.h"
#include "Scene.h"

namespace spark
{
Component::Component(std::string&& name) : name(std::move(name)) {}

Component::~Component()
{
    SPARK_TRACE("Component '{}' destroyed!", name);
}

void Component::drawUI()
{
    beginDrawingWindow();
    if(active)
    {
        drawUIBody();
    }
    removeComponentFromGameObjectButton();
    endDrawingWindow();
}

void Component::beginDrawingWindow()
{
    ImGui::PushID(this);
    ImGui::BeginGroupPanel(name.c_str(), {-1, 0});
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

void Component::removeComponentFromGameObjectButton()
{
    if(ImGui::Button("Delete"))
    {
        removeComponent();
    }
}

void Component::endDrawingWindow()
{
    ImGui::EndGroupPanel();
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
    if(!gameObject.expired())
    {
        gameObject.lock()->removeComponent(shared_from_this());
    }

    gameObject = game_object;
    if(game_object)
    {
        game_object->addComponent(shared_from_this());
    }
}

void Component::setActive(bool active_)
{
    active = active_;
    if(active)
    {
        onActive();
    }
    else
    {
        onInactive();
    }
}

void Component::removeComponent()
{
    if(const auto go = getGameObject(); go)
    {
        go->removeComponent(shared_from_this());
    }
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