#pragma once

#include <optional>

#include "Component.h"
#include "GameObject.h"
#include "Resource.h"

namespace spark
{
namespace resources
{
    class Texture;
    class Model;
};  // namespace resources

class SparkGui
{
    public:
    void drawGui();

    SparkGui() = default;
    ~SparkGui() = default;

    static std::shared_ptr<Component> addComponent();
    static std::optional<std::shared_ptr<resources::Model>> selectModelByFilePicker();
    static std::optional<std::shared_ptr<resources::Texture>> selectTextureByFilePicker();
    static std::optional<std::shared_ptr<Scene>> selectSceneByFilePicker();
    static std::filesystem::path getRelativePathToSaveSceneByFilePicker();

    template<typename T>
    static std::optional<T> getDraggedObject(std::string&& payloadName);

    template<typename T>
    static std::shared_ptr<T> getObject(const std::string&& variableName, std::shared_ptr<T> object);

    private:
    void drawMainMenuGui();
    void drawSparkSettings(bool* p_open);
    int checkCurrentItem(const char** items) const;
    static std::string putExtensionsInOneStringSeparatedByCommas(const std::vector<std::string>& fileExtensions);
    static std::shared_ptr<resourceManagement::Resource> getResourceIdentifierByFilePicker(const char* buttonName,
                                                                                           const std::vector<std::string>& fileExtensions);

    const static std::map<std::string, std::function<std::shared_ptr<Component>()>> componentCreation;
};

template<typename T>
std::optional<T> SparkGui::getDraggedObject(std::string&& payloadName)
{
    /*if (object == nullptr)
    {
        std::string objectName = variableName + ": " + "nullptr";
        ImGui::Text(objectName.c_str());
    }

    if (object != nullptr)
    {
        std::string objectName = variableName + ": " + "(" + typeid(object).name() + ")";
        ImGui::Text(objectName.c_str());
    }*/

    if(ImGui::BeginDragDropTarget())
    {
        if(const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(payloadName.c_str()))
        {
            IM_ASSERT(payload->DataSize == sizeof(T));
            T draggedObject = *static_cast<const T*>(payload->Data);
            return draggedObject;
        }
        ImGui::EndDragDropTarget();
    }
    return std::nullopt;
}

template<typename T>
std::shared_ptr<T> SparkGui::getObject(const std::string&& variableName, std::shared_ptr<T> object)
{
    if(object == nullptr)
    {
        std::string objectName = variableName + ": " + "nullptr";
        ImGui::Text(objectName.c_str());
    }

    if(object != nullptr)
    {
        if(Component* c = dynamic_cast<Component*>(object.get()); c != nullptr)
        {
            std::string objectName = variableName + ": " + c->getName() + " (" + typeid(c).name() + ")";
            ImGui::Text(objectName.c_str());
        }
        if(GameObject* g = dynamic_cast<GameObject*>(object.get()); g != nullptr)
        {
            std::string objectName = variableName + ": " + g->getName() + " (" + typeid(g).name() + ")";
            ImGui::Text(objectName.c_str());
        }
    }

    if(ImGui::BeginDragDropTarget())
    {
        if(const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("OBJECT_DRAG_AND_DROP"))
        {
            IM_ASSERT(payload->DataSize == sizeof(std::shared_ptr<GameObject>));
            std::shared_ptr<GameObject> gameObject = *static_cast<const std::shared_ptr<GameObject>*>(payload->Data);
            // TODO: resolve this
            // if(dynamic_cast<T*>(gameObject.get()) != nullptr)
            //{
            //	return std::dynamic_pointer_cast<T>(gameObject);
            //}

            return gameObject->getComponent<T>();  // return nullptr if there is not any type T component
        }
        ImGui::EndDragDropTarget();
    }
    return object;
}
}  // namespace spark