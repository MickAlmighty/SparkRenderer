#ifndef COMPONENT_H
#define COMPONENT_H

#include "EngineSystems/SceneManager.h"
#include "Scene.h"

#include <rttr/registration_friend>
#include <rttr/registration>
#include <GUI/ImGui/imgui.h>

namespace spark
{
class GameObject;
class Component abstract : public std::enable_shared_from_this<Component>
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
    virtual void setActive(bool active_);
    template<class T>
    void removeComponent()
    {
        auto remove = [component = shared_from_base<T>()]() { component->getGameObject()->removeComponent<T>(component); };
        getGameObject()->getScene()->toRemove.push_back(remove);
    }
    template<class T>
    void removeComponentGUI()
    {
        if(ImGui::Button("Delete"))
        {
            removeComponent<T>();
        }
    }

    protected:
    bool active{true};
    template<typename Derived>
    std::shared_ptr<Derived> shared_from_base()
    {
        return std::static_pointer_cast<Derived>(shared_from_this());
    }
    std::shared_ptr<Component> getComponentPtr();

    private:
    std::string name{"Component"};
    std::weak_ptr<GameObject> gameObject;
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE();
};
}  // namespace spark
#endif