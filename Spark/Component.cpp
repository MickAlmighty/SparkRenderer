#include "Component.h"

#include "GUI/ImGui/imgui_custom_widgets.h"
#include "GameObject.h"
#include "Logging.h"
#include "Scene.h"

namespace spark
{
Component::~Component()
{
    SPARK_TRACE("Component '{}' destroyed!", getName());
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
    ImGui::BeginGroupPanel(getName().c_str(), {-1, 0});

    bool currentActive = getActive();
    ImGui::Checkbox("Enabled", &currentActive);
    if(currentActive != getActive())
    {
        setActive(currentActive);
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
    return get_type().get_name().begin();
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

std::shared_ptr<Component> Component::getSharedPtrBase()
{
    return std::static_pointer_cast<Component>(shared_from_this());
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::Component>("Component")
        .method("getSharedPtrBase", &spark::Component::getSharedPtrBase)
        .property("active", &spark::Component::active)
        .property("gameObject", &spark::Component::getGameObject, &spark::Component::setGameObject, rttr::registration::public_access);
}