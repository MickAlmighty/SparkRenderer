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
    virtual ~Component();
    Component(Component&) = delete;
    Component(Component&&) = delete;
    Component& operator=(const Component&) = delete;
    Component& operator=(Component&&) = delete;

    virtual void update() = 0;
    void drawUI();

    void setGameObject(const std::shared_ptr<GameObject> game_object);
    std::shared_ptr<GameObject> getGameObject() const;
    std::string getName() const;
    bool getActive() const;
    void setActive(bool active_);

    protected:
    virtual void drawUIBody() {}

    std::shared_ptr<Component> getSharedPtrBase();

    private:
    virtual void onActive() {}
    virtual void onInactive() {}

    void beginDrawingWindow();
    void removeComponentFromGameObjectButton();
    void removeComponent();
    void endDrawingWindow();

    std::weak_ptr<GameObject> gameObject;
    bool active{true};
    friend class GameObject;
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE();
};
}  // namespace spark