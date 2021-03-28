#pragma once

#include <list>
#include <memory>

#include "Structs.h"
#include "Component.h"

namespace spark
{
class GameObject final : public std::enable_shared_from_this<GameObject>
{
    public:
    GameObject() = default;
    explicit GameObject(std::string&& name);
    ~GameObject();
    GameObject(GameObject&) = delete;
    GameObject(GameObject&&) = delete;
    GameObject& operator=(const GameObject&) = delete;
    GameObject& operator=(GameObject&&) = delete;

    std::shared_ptr<GameObject> getParent() const;
    std::shared_ptr<Scene> getScene() const;
    void setParent(const std::shared_ptr<GameObject> newParent);
    void setScene(const std::shared_ptr<Scene> newScene);
    void addChild(const std::shared_ptr<GameObject>& newChild, const std::shared_ptr<GameObject>& parent);
    void addComponent(const std::shared_ptr<Component>& component);
    bool removeChild(std::string&& gameObjectName);
    bool removeChild(std::shared_ptr<GameObject> child);
    void drawGUI();

    std::string getName() const;
    bool isActive() const;
    bool isStatic() const;

    void setActive(bool active_);
    void setStatic(bool static_);

    template<class T>
    std::shared_ptr<T> getComponent();

    template<class T>
    std::shared_ptr<T> getAllComponentsOfType();

    template<class T>
    bool removeComponent(const std::string& name);

    template<class T>
    bool removeComponent(const std::shared_ptr<T>& c);

    template<class T>
    bool removeComponentOfType();

    template<class T>
    void removeAllComponentsOfType();

    Transform transform;

    private:
    friend class Scene;
    std::string name{"GameObject"};
    bool active{true};
    bool staticObject{false};
    std::weak_ptr<Scene> scene;
    std::weak_ptr<GameObject> parent;
    std::vector<std::shared_ptr<GameObject>> children;
    std::vector<std::shared_ptr<Component>> components;
    void update();
    void fixedUpdate();
    void drawGizmos();
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE();
};

template<class T>
std::shared_ptr<T> GameObject::getComponent()
{
    // TODO: add RTTR reflection checks in template methods of GameObject, they'll make the code much cleaner
    auto component_it = std::find_if(std::begin(components), std::end(components), [](const std::shared_ptr<Component>& component) {
        T* comp_ptr = dynamic_cast<T*>(component.get());
        return comp_ptr != nullptr;
    });
    if(component_it != components.end())
    {
        return std::dynamic_pointer_cast<T>(*component_it);
    }
    return nullptr;
}

template<class T>
std::shared_ptr<T> GameObject::getAllComponentsOfType()
{
    std::vector<std::shared_ptr<T>> components;

    for(const auto& component : components)
    {
        const auto componentOfTypeT = std::dynamic_pointer_cast<T>(component);
        if (componentOfTypeT != nullptr)
        {
            components.push_back(componentOfTypeT);
        }
    }

    return components;
}

template<class T>
bool GameObject::removeComponent(const std::string& name)
{
    auto component_it = std::find_if(std::begin(components), std::end(components), [&name](const std::shared_ptr<Component>& component) {
        if(dynamic_cast<T*>(component.get()))
        {
            return component->getName() == name;
        }
        return false;
    });
    if(component_it != components.end())
    {
        components.erase(component_it);
        return true;
    }
    return false;
}

template<class T>
bool GameObject::removeComponent(const std::shared_ptr<T>& c)
{
    auto component_it = std::find_if(std::begin(components), std::end(components), [&c](const std::shared_ptr<Component>& component) {
        if(dynamic_cast<T*>(component.get()))
        {
            return component.get() == c.get();
        }
        return false;
    });
    if(component_it != components.end())
    {
        components.erase(component_it);
        return true;
    }
    return false;
}

template<class T>
bool GameObject::removeComponentOfType()
{
    // TODO: add RTTR reflection checks in template methods of GameObject, they'll make the code much cleaner
    auto component_it = std::find_if(std::begin(components), std::end(components), [](const std::shared_ptr<Component>& component) {
        T* comp_ptr = dynamic_cast<T*>(component.get());
        return comp_ptr != nullptr;
    });
    if(component_it != components.end())
    {
        components.erase(component_it);
        return true;
    }
    return false;
}

template<class T>
void GameObject::removeAllComponentsOfType()
{
    for(auto it = components.begin(); it != components.end();)
    {
        T* component_ptr = dynamic_cast<T*>(*it);
        if(component_ptr != nullptr)
        {
            it = components.erase(it);
            continue;
        }
        ++it;
    }
}
}  // namespace spark