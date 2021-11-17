#pragma once

#include <memory>

#include "Component.h"
#include "Transform.hpp"

namespace spark
{
class Scene;

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

    void drawGUI();

    std::string getName() const;
    bool isActive() const;
    bool isStatic() const;

    void setActive(bool active_);
    void setStatic(bool static_);
    std::shared_ptr<GameObject> getParent() const;
    void setParent(const std::shared_ptr<GameObject> newParent);
    Scene* getScene() const;
    void setScene(Scene* newScene);

    const std::vector<std::shared_ptr<GameObject>>& getChildren() const;
    void addChild(const std::shared_ptr<GameObject>& newChild);
    bool removeChild(const std::shared_ptr<GameObject>& child);
    bool removeChild(GameObject* child);

    template<typename T>
    void addComponent();

    void addComponent(const std::shared_ptr<Component>& component);

    template<class T>
    std::shared_ptr<T> getComponent();

    template<class T>
    std::vector<std::shared_ptr<T>> getAllComponentsOfType();

    bool removeComponent(const std::shared_ptr<Component>& c);
    bool removeComponent(const std::string& componentName);

    template<class T>
    bool removeComponent();

    template<class T>
    void removeAllComponentsOfType();

    Transform transform;

    private:
    void update();
    void drawGizmos();
    void setSceneRecursive(Scene* newScene);

    std::weak_ptr<GameObject> parent;
    std::string name{"GameObject"};
    std::vector<std::shared_ptr<GameObject>> children;
    std::vector<std::shared_ptr<Component>> components;
    Scene* scene{nullptr};
    bool active{true};
    bool staticObject{false};

    friend class Scene;
    RTTR_REGISTRATION_FRIEND;
};

template<typename T>
void GameObject::addComponent()
{
    const auto component = std::make_shared<T>();
    component->gameObject = shared_from_this();
    components.emplace_back(component);
}

template<class T>
std::shared_ptr<T> GameObject::getComponent()
{
    // TODO: add RTTR reflection checks in template methods of GameObject, they'll make the code much cleaner
    auto component_it = std::find_if(std::begin(components), std::end(components),
                                     [](const std::shared_ptr<Component>& component) { return dynamic_cast<T*>(component.get()); });
    if(component_it != components.end())
    {
        return std::static_pointer_cast<T>(*component_it);
    }
    return nullptr;
}

template<>
inline std::vector<std::shared_ptr<Component>> GameObject::getAllComponentsOfType()
{
    return components;
}

template<class T>
std::vector<std::shared_ptr<T>> GameObject::getAllComponentsOfType()
{
    std::vector<std::shared_ptr<T>> componentsOfTypeT;

    for(const auto& component : components)
    {
        if(const auto componentOfTypeT = std::dynamic_pointer_cast<T>(component); componentOfTypeT != nullptr)
        {
            componentsOfTypeT.push_back(componentOfTypeT);
        }
    }

    return componentsOfTypeT;
}

template<class T>
bool GameObject::removeComponent()
{
    // TODO: add RTTR reflection checks in template methods of GameObject, they'll make the code much cleaner
    auto component_it = std::find_if(std::begin(components), std::end(components),
                                     [](const std::shared_ptr<Component>& component) { return dynamic_cast<T*>(component.get()); });
    if(component_it != components.end())
    {
        (*component_it)->gameObject.reset();
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
        if(T* component_ptr = dynamic_cast<T*>(it->get()); component_ptr != nullptr)
        {
            component_ptr->gameObject.reset();
            it = components.erase(it);
            continue;
        }
        ++it;
    }
}
}  // namespace spark