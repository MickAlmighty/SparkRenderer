#include "SparkGui.h"

#include "Animation.hpp"
#include "utils/CommonUtils.h"
#include "ImGuizmo.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"
#include "JsonSerializer.h"
#include "Model.h"
#include "ResourceLibrary.h"
#include "Spark.h"
#include "Texture.h"

namespace spark
{
void SparkGui::drawGui()
{
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();

    bool show = true;
    ImGui::ShowDemoWindow(&show);
    if(ImGui::BeginMainMenuBar())
    {
        drawMainMenuGui();
        Spark::get().getSceneManager().drawGui();
        Spark::get().drawGui();
        ImGui::EndMainMenuBar();
    }

    PUSH_DEBUG_GROUP(IMGUI)
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    POP_DEBUG_GROUP()
}

void SparkGui::drawMainMenuGui()
{
    static bool showEngineSettings = false;
    static bool animationCreator = false;
    if(ImGui::BeginMenu("Engine"))
    {
        ImGui::MenuItem("Spark Settings", NULL, &showEngineSettings);
        ImGui::MenuItem("Animation Creator", NULL, &animationCreator);
        ImGui::Separator();
        if(ImGui::MenuItem("Exit", "Esc"))
        {
            Spark::get().getRenderingContext().closeWindow();
        }
        ImGui::EndMenu();
    }

    if(showEngineSettings)
        drawSparkSettings(&showEngineSettings);

    if(animationCreator)
        drawAnimationCreatorWindow(&animationCreator);
}

void SparkGui::drawSparkSettings(bool* p_open)
{
    if(!ImGui::Begin("Spark Settings", p_open, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::End();
        return;
    }
    ImGui::Text("Framerate: %f", ImGui::GetIO().Framerate);
    bool vsync = Spark::get().vsync;
    ImGui::Checkbox("V-Sync", &vsync);

    if(vsync != Spark::get().vsync)
    {
        Spark::get().vsync = vsync;
        Spark::get().getRenderingContext().setVsync(vsync);
    }

    static const char* items[4] = {"1280x720", "1600x900", "1920x1080", "1920x1055"};
    static int current_item = checkCurrentItem(items);
    if(ImGui::Combo("Resolution", &current_item, items, IM_ARRAYSIZE(items)))
    {
        if(current_item == 0)
        {
            Spark::get().getRenderingContext().resizeWindow(1280, 720);
        }
        else if(current_item == 1)
        {
            Spark::get().getRenderingContext().resizeWindow(1600, 900);
        }
        else if(current_item == 2)
        {
            Spark::get().getRenderingContext().resizeWindow(1920, 1080);
        }
        else if(current_item == 3)
        {
            Spark::get().getRenderingContext().resizeWindow(1920, 1055);
        }
    }

    ImGui::End();
}

void SparkGui::drawAnimationCreatorWindow(bool* p_open)
{
    if(!ImGui::Begin("Animation Creator", p_open, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::End();
        return;
    }

    Spark::get().getAnimationCreator().drawGui();

    ImGui::End();
}

int SparkGui::checkCurrentItem(const char** items) const
{
    const std::string resolution =
        std::to_string(Spark::get().getRenderingContext().width) + "x" + std::to_string(Spark::get().getRenderingContext().height);
    for(int i = 0; i < 4; i++)
    {
        std::string item(items[i]);
        if(item == resolution)
            return i;
    }
    return 0;
}

std::optional<std::string> SparkGui::addComponent()
{
    std::optional<std::string> componentTypeNameOpt{std::nullopt};
    if(ImGui::Button("Add Component"))
    {
        ImGui::OpenPopup("Components");
    }

    if(ImGui::BeginPopupModal("Components", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        for(auto& type : rttr::type::get_types())
        {
            // SPARK_INFO("{}", type.get_name().begin());
            if(type.is_wrapper())
            {
                if(auto rawType = type.get_wrapped_type().get_raw_type();
                   rawType.is_derived_from<Component>() && rawType != rttr::type::get<Component>())
                {
                    std::string componentTypeName{rawType.get_name().begin()};
                    if(ImGui::Button(componentTypeName.c_str()))
                    {
                        componentTypeNameOpt = componentTypeName;
                        ImGui::CloseCurrentPopup();
                        break;
                    }
                }
            }
        }
        if(ImGui::Button("Close"))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    return componentTypeNameOpt;
}

std::optional<std::shared_ptr<resources::Model>> SparkGui::selectModelByFilePicker()
{
    constexpr auto buttonName = "Select Model";
    const auto fileExtensions = resourceManagement::ResourceFactory::supportedModelExtensions();

    if(const auto resource = getResourceIdentifierByFilePicker(buttonName, fileExtensions); resource)
        return {std::static_pointer_cast<resources::Model>(resource)};

    return std::nullopt;
}

std::optional<std::shared_ptr<resources::Texture>> SparkGui::selectTextureByFilePicker()
{
    constexpr auto buttonName = "Select Texture";
    const auto fileExtensions = resourceManagement::ResourceFactory::supportedTextureExtensions();

    if(const auto resource = getResourceIdentifierByFilePicker(buttonName, fileExtensions); resource)
        return {std::static_pointer_cast<resources::Texture>(resource)};

    return std::nullopt;
}

std::optional<std::shared_ptr<resources::AnimationData>> SparkGui::selectAnimationDataByFilePicker()
{
    constexpr auto buttonName = "Select Animation";
    const auto fileExtensions = resourceManagement::ResourceFactory::supportedAnimationExtensions();

    if(const auto resource = getResourceIdentifierByFilePicker(buttonName, fileExtensions); resource)
        return {std::static_pointer_cast<resources::AnimationData>(resource)};

    return std::nullopt;
}

std::optional<std::shared_ptr<Scene>> SparkGui::selectSceneByFilePicker()
{
    constexpr auto buttonName = "Select Scene";
    const auto fileExtensions = resourceManagement::ResourceFactory::supportedSceneExtensions();

    if(const auto resource = getResourceIdentifierByFilePicker(buttonName, fileExtensions); resource)
        return {std::static_pointer_cast<Scene>(resource)};

    return std::nullopt;
}

std::filesystem::path SparkGui::getRelativePathToSaveSceneByFilePicker()
{
    if(ImGui::Button("Save Scene"))
    {
        ImGui::OpenPopup("Save File");
    }

    std::filesystem::path filepath;
    const auto extensions = putExtensionsInOneStringSeparatedByCommas(resourceManagement::ResourceFactory::supportedSceneExtensions());

    if(file_dialog.showFileDialog("Save File", imgui_addons::ImGuiFileBrowser::DialogMode::SAVE, ImVec2(700, 310), extensions))
    {
        filepath = file_dialog.selected_path;
    }

    return filepath;
}

std::filesystem::path SparkGui::getRelativePathToSaveAnimationByFilePicker()
{
    if(ImGui::Button("Save Animation"))
    {
        ImGui::OpenPopup("Save File");
    }

    std::filesystem::path filepath;
    const auto extensions = putExtensionsInOneStringSeparatedByCommas(resourceManagement::ResourceFactory::supportedAnimationExtensions());

    if(file_dialog.showFileDialog("Save File", imgui_addons::ImGuiFileBrowser::DialogMode::SAVE, ImVec2(700, 310), extensions))
    {
        filepath = file_dialog.selected_path;
    }

    return filepath;
}

void SparkGui::setFilePickerPath(const std::string& path)
{
    file_dialog = imgui_addons::ImGuiFileBrowser(path);
}

std::shared_ptr<resourceManagement::Resource> SparkGui::getResourceIdentifierByFilePicker(const char* buttonName,
                                                                                          const std::vector<std::string>& fileExtensions)
{
    if(ImGui::Button(buttonName))
    {
        ImGui::OpenPopup("Select File");
    }

    const auto extensions = putExtensionsInOneStringSeparatedByCommas(fileExtensions);

    if(file_dialog.showFileDialog("Select File", imgui_addons::ImGuiFileBrowser::DialogMode::OPEN, ImVec2(700, 310), extensions))
    {
        if(const std::filesystem::path filepath = file_dialog.selected_path; !filepath.empty())
        {
            return Spark::get().getResourceLibrary().getResourceByFullPath<resourceManagement::Resource>(filepath);
        }
    }

    return nullptr;
}

std::string SparkGui::putExtensionsInOneStringSeparatedByCommas(const std::vector<std::string>& fileExtensions)
{
    std::stringstream ss;
    for(size_t i = 0; i < fileExtensions.size(); ++i)
    {
        ss << fileExtensions[i];
        if(i < fileExtensions.size() - 1)
            ss << ",";
    }

    return ss.str();
}

}  // namespace spark
