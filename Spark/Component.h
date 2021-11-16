#pragma once

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
    void drawUI();

    void endDrawingWindow();
    void setGameObject(const std::shared_ptr<GameObject> game_object);
    std::shared_ptr<GameObject> getGameObject() const;
    std::string getName() const;
    bool getActive() const;
    void setActive(bool active_);

    protected:
    virtual void drawUIBody(){};

    template<typename Derived>
    std::shared_ptr<Derived> shared_from_base();
    std::shared_ptr<Component> getComponentPtr();

    private:
    virtual void onActive() {}
    virtual void onInactive() {}

    void beginDrawingWindow();
    void removeComponentFromGameObjectButton();
    void removeComponent();

    bool active{true};
    std::string name{"Component"};
    std::weak_ptr<GameObject> gameObject;
    friend class GameObject;
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE();
};

template<typename Derived>
std::shared_ptr<Derived> Component::shared_from_base()
{
    return std::static_pointer_cast<Derived>(shared_from_this());
}
}  // namespace spark