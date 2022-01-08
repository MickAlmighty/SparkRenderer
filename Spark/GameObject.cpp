#include "GameObject.h"

#include <glm/gtc/type_ptr.hpp>

#include "EditorCamera.hpp"
#include "Component.h"
#include "GUI/ImGui/imgui.h"
#include "GUI/ImGuizmo.h"
#include "Scene.h"
#include "GUI/SparkGui.h"
#include "Logging.h"

namespace spark
{
void GameObject::update()
{
    if(parent.lock() == nullptr)
    {
        transform.world.setMatrix(transform.local.getMatrix());
    }
    else
    {
        transform.world.setMatrix(parent.lock()->transform.world.getMatrix() * transform.local.getMatrix());
    }

    for(const auto& component : components)
    {
        if(component->getActive())
        {
            component->update();
        }
    }

    for(const auto& child : children)
    {
        if(child->isActive())
        {
            child->update();
        }
    }
}

std::shared_ptr<GameObject> GameObject::getParent() const
{
    return parent.lock();
}

Scene* GameObject::getScene() const
{
    return scene;
}

void GameObject::setParent(const std::shared_ptr<GameObject> newParent)
{
    if(const auto currentParent = parent.lock(); newParent && newParent != currentParent && newParent.get() != this)
    {
        if(currentParent)
        {
            currentParent->removeChild(this);
        }

        newParent->children.push_back(shared_from_this());
        scene = newParent->scene;
        parent = newParent;
    }
}

void GameObject::setScene(Scene* newScene)
{
    this->scene = newScene;
}

void GameObject::addChild(const std::shared_ptr<GameObject>& newChild)
{
    if(!newChild)
    {
        return;
    }

    if(const auto gameObjectIt = std::find_if(std::begin(children), std::end(children),
                                              [&newChild](const std::shared_ptr<GameObject>& gameObject) { return gameObject == newChild; });
       gameObjectIt != children.end())
    {
        return;
    }

    const auto setParentFor = [newParent = shared_from_this()](const std::shared_ptr<GameObject>& newChild)
    {
        if(const auto currentParent = newChild->parent.lock(); currentParent != newParent)
        {
            if(currentParent)
            {
                currentParent->removeChild(newChild);
            }

            newChild->parent = newParent;
        }
    };

    setParentFor(newChild);
    newChild->setScene(scene);
    children.push_back(newChild);
}

bool GameObject::removeChild(const std::shared_ptr<GameObject>& child)
{
    return removeChild(child.get());
}

bool GameObject::removeChild(GameObject* child)
{
    const auto gameObjectIt = std::find_if(std::begin(children), std::end(children),
                                           [&child](const std::shared_ptr<GameObject>& gameObject) { return gameObject.get() == child; });
    if(gameObjectIt != children.end())
    {
        (*gameObjectIt)->parent.reset();
        children.erase(gameObjectIt);
        return true;
    }
    return false;
}

std::shared_ptr<Component> GameObject::addComponent(const char* componentTypeName)
{
    if(componentTypeName)
    {
        return addComponent(componentTypeName);
    }

    return nullptr;
}

std::shared_ptr<Component> GameObject::addComponent(const std::string& componentTypeName)
{
    std::shared_ptr<Component> component{nullptr};

    if(const auto type = rttr::type::get_by_name(componentTypeName); type.is_derived_from(rttr::type::get<Component>()))
    {
        const auto constructComponentByTypeName = [&type](const rttr::variant& variant)
        {
            const rttr::method convMethod{type.get_method("getSharedPtrBase")};

            const rttr::variant wrappedVal{variant.extract_wrapped_value()};
            return convMethod.invoke(wrappedVal).get_value<std::shared_ptr<Component>>();
        };

        if(const auto variant = type.create(); variant.is_valid())
        {
            component = constructComponentByTypeName(variant);
            component->setGameObject(shared_from_this());
            components.push_back(component);
        }
    }

    return component;
}

const std::vector<std::shared_ptr<Component>>& GameObject::getComponents() const
{
    return components;
}

bool GameObject::removeComponent(const std::shared_ptr<Component>& c)
{
    const auto componentIt =
        std::find_if(std::begin(components), std::end(components), [&c](const std::shared_ptr<Component>& component) { return component == c; });
    if(componentIt != components.end())
    {
        (*componentIt)->gameObject.reset();
        components.erase(componentIt);
        return true;
    }
    return false;
}

bool GameObject::removeComponent(const std::string& componentTypeName)
{
    const auto componentIt =
        std::find_if(std::begin(components), std::end(components),
                     [&componentTypeName](const std::shared_ptr<Component>& component) { return component->getName() == componentTypeName; });
    if(componentIt != components.end())
    {
        (*componentIt)->gameObject.reset();
        components.erase(componentIt);
        return true;
    }
    return false;
}

void GameObject::drawGUI()
{
    ImGui::Text(name.c_str());
    static char nameInput[64] = "";
    ImGui::InputTextWithHint("", name.c_str(), nameInput, 64, ImGuiInputTextFlags_CharsNoBlank);
    ImGui::SameLine();

    if(ImGui::Button("Change Name") && nameInput[0] != '\0')
    {
        name = nameInput;
        std::fill_n(nameInput, 64, '\0');
    }

    ImGui::Checkbox("active", &active);
    ImGui::SameLine();
    ImGui::Checkbox("static", &staticObject);

    transform.local.drawGUI();
    drawGizmos();
    for(auto& component : components)
        component->drawUI();

    ImGui::NewLine();
    if(const auto componentNameOpt = SparkGui::addComponent(); componentNameOpt.has_value())
    {
        addComponent(componentNameOpt.value());
    }
}

std::string GameObject::getName() const
{
    return name;
}

bool GameObject::isActive() const
{
    return active;
}

bool GameObject::isStatic() const
{
    return staticObject;
}

void GameObject::setActive(bool active_)
{
    active = active_;

    for(const auto& child : children)
    {
        child->setActive(active);
    }
}

void GameObject::setStatic(bool static_)
{
    staticObject = static_;
}

const std::vector<std::shared_ptr<GameObject>>& GameObject::getChildren() const
{
    return children;
}

void GameObject::drawGizmos()
{
    static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::ROTATE);
    static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::WORLD);
    if(ImGui::IsKeyPressed(GLFW_KEY_T))
        mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
    if(ImGui::IsKeyPressed(GLFW_KEY_R))
        mCurrentGizmoOperation = ImGuizmo::ROTATE;
    if(ImGui::IsKeyPressed(GLFW_KEY_Y))
    {
        mCurrentGizmoOperation = ImGuizmo::SCALE;
        mCurrentGizmoMode = ImGuizmo::LOCAL;
    }

    if(ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
        mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
    ImGui::SameLine();
    if(ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
        mCurrentGizmoOperation = ImGuizmo::ROTATE;
    ImGui::SameLine();
    if(ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
        mCurrentGizmoOperation = ImGuizmo::SCALE;

    glm::mat4 worldMatrix = getParent()->transform.world.getMatrix() * transform.local.getMatrix();  // transform.world.getMatrix();

    if(mCurrentGizmoOperation != ImGuizmo::SCALE)
    {
        if(ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
            mCurrentGizmoMode = ImGuizmo::LOCAL;
        ImGui::SameLine();
        if(ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
            mCurrentGizmoMode = ImGuizmo::WORLD;
    }

    const ImGuiIO& io = ImGui::GetIO();
    const auto camera = getScene()->editorCamera;
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
    ImGuizmo::Manipulate(glm::value_ptr(camera->getViewMatrix()), glm::value_ptr(camera->getProjection()), mCurrentGizmoOperation, mCurrentGizmoMode,
                         &worldMatrix[0][0]);

    glm::mat4 localMatrix = glm::inverse(getParent()->transform.world.getMatrix()) *
                            worldMatrix;  // getting new localTransform by multiplying by inverse of parent world transform

    glm::vec3 pos{}, scale{}, rotation{};
    ImGuizmo::DecomposeMatrixToComponents(&localMatrix[0][0], &pos.x, &rotation.x, &scale.x);

    if(glm::length(pos - transform.local.getPosition()) > 0.1f)
    {
        transform.local.setPosition(pos);
    }
    if(glm::length(scale - transform.local.getScale()) > 0.1f)
    {
        transform.local.setScale(scale);
    }
    if(glm::length(rotation - transform.local.getRotationDegrees()) > 0.1f)
    {
        transform.local.setRotationDegrees(rotation);
    }
}

GameObject::GameObject(std::string&& name) : name(std::move(name)) {}

GameObject::~GameObject()
{
    SPARK_TRACE("GameObject '{}' destroyed!", name);
}

}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::GameObject>("GameObject")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("transform", &spark::GameObject::transform)
        .property("name", &spark::GameObject::name)
        .property("active", &spark::GameObject::active)
        .property("staticObject", &spark::GameObject::staticObject)
        .property("scene", &spark::GameObject::getScene, &spark::GameObject::setScene,
                  rttr::registration::public_access)  //(rttr::detail::metadata(spark::SerializerMeta::Serializable, false))
        .property("parent", &spark::GameObject::getParent, &spark::GameObject::setParent, rttr::registration::public_access)
        .property("children", &spark::GameObject::children)
        .property("components", &spark::GameObject::components);
}