#pragma once

#include "GUI/ImGui/imgui.h"

#include <rttr/registration_friend>
#include <rttr/registration>

namespace spark
{
class GameObject;
class Component : public std::enable_shared_from_this<Component>
{
    public:
    Component() = default;
    explicit Component(std::string&& name);
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
    void setGameObject(const std::shared_ptr<GameObject> game_object);
    std::shared_ptr<GameObject> getGameObject() const;
    std::string getName() const;
    bool getActive() const;
    void setActive(bool active_);
    void removeComponent();

    template<class T>
    void removeComponentGUI()
    {
        if(ImGui::Button("Delete"))
        {
            removeComponent();
        }
    }

    protected:
    template<typename Derived>
    std::shared_ptr<Derived> shared_from_base()
    {
        return std::static_pointer_cast<Derived>(shared_from_this());
    }
    std::shared_ptr<Component> getComponentPtr();

    private:
    virtual void onActive() {}
    virtual void onInactive() {}

    bool active{true};
    std::string name{"Component"};
    std::weak_ptr<GameObject> gameObject;
    friend class GameObject;
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE();
};
}  // namespace spark