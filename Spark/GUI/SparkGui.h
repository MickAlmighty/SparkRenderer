#pragma once

#include <optional>

#include "Component.h"
#include "GameObject.h"
#include "ImGuiFileBrowser.h"
#include "Resource.h"

namespace spark
{
namespace resources
{
    class AnimationData;
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
    static std::optional<std::shared_ptr<resources::AnimationData>> SparkGui::selectAnimationDataByFilePicker();
    static std::optional<std::shared_ptr<Scene>> selectSceneByFilePicker();
    static std::filesystem::path getRelativePathToSaveSceneByFilePicker();
    static std::filesystem::path getRelativePathToSaveAnimationByFilePicker();
    static void setFilePickerPath(const std::string& path);

    private:
    void drawMainMenuGui();
    void drawSparkSettings(bool* p_open);
    void drawAnimationCreatorWindow(bool* p_open);
    int checkCurrentItem(const char** items) const;
    static std::string putExtensionsInOneStringSeparatedByCommas(const std::vector<std::string>& fileExtensions);
    static std::shared_ptr<resourceManagement::Resource> getResourceIdentifierByFilePicker(const char* buttonName,
                                                                                           const std::vector<std::string>& fileExtensions);

    const static std::map<std::string, std::function<std::shared_ptr<Component>()>> componentCreation;
    inline static imgui_addons::ImGuiFileBrowser file_dialog{};
};
}  // namespace spark