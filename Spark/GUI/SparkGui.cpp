#include "GUI/SparkGui.h"

#include "EngineSystems/SparkRenderer.h"
#include "ImGuizmo.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"
#include "JsonSerializer.h"
#include "Lights/DirectionalLight.h"
#include "Lights/LightProbe.h"
#include "Lights/PointLight.h"
#include "Lights/SpotLight.h"
#include "Mesh.h"
#include "MeshPlane.h"
#include "Model.h"
#include "ModelMesh.h"
#include "ResourceLibrary.h"
#include "ResourceLoader.h"
#include "Skybox.h"
#include "Spark.h"
#include "Texture.h"

namespace spark
{
const std::map<std::string, std::function<std::shared_ptr<Component>()>> SparkGui::componentCreation{
    // TODO: replace with a reflection-based list
    {"ModelMesh", [] { return std::make_shared<ModelMesh>(); }},
    {"MeshPlane", [] { return std::make_shared<MeshPlane>(); }},
    {"DirectionalLight", [] { return std::make_shared<DirectionalLight>(); }},
    {"PointLight", [] { return std::make_shared<PointLight>(); }},
    {"SpotLight", [] { return std::make_shared<SpotLight>(); }},
    {"LightProbe", [] { return std::make_shared<LightProbe>(); }},
    {"Skybox", [] { return std::make_shared<Skybox>(); }}};

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
        SceneManager::getInstance()->drawGui();
        SparkRenderer::getInstance()->drawGui();
        ImGui::EndMainMenuBar();
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void SparkGui::drawMainMenuGui()
{
    static bool showEngineSettings = false;
    if(ImGui::BeginMenu("Engine"))
    {
        ImGui::MenuItem("Spark Settings", NULL, &showEngineSettings);
        ImGui::Separator();
        if(ImGui::MenuItem("Exit", "Esc"))
        {
            Spark::oglContext.closeWindow();
        }
        ImGui::EndMenu();
    }

    if(showEngineSettings)
        drawSparkSettings(&showEngineSettings);
}

void SparkGui::drawSparkSettings(bool* p_open)
{
    if(!ImGui::Begin("Spark Settings", p_open, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::End();
        return;
    }
    /*static char buf1[128];
    static char buf2[128];
    ImGui::InputTextWithHint("Path to Models", Spark::pathToModelMeshes.string().c_str(), buf1, 128);
    ImGui::InputTextWithHint("Path to Resources", Spark::pathToResources.string().c_str(), buf2, 128);*/
    ImGui::Text("Framerate: %f", ImGui::GetIO().Framerate);
    ImGui::Text("Path to models:");
    ImGui::SameLine();
    ImGui::Text(Spark::pathToModelMeshes.string().c_str());
    ImGui::Text("Path to resources:");
    ImGui::SameLine();
    ImGui::Text(Spark::pathToResources.string().c_str());
    bool vsync = Spark::vsync;
    ImGui::Checkbox("V-Sync", &vsync);

    if(vsync != Spark::vsync)
    {
        Spark::vsync = vsync;
        Spark::oglContext.setVsync(vsync);
    }

    static const char* items[4] = {"1280x720", "1600x900", "1920x1080", "1920x1055"};
    static int current_item = checkCurrentItem(items);
    if(ImGui::Combo("Resolution", &current_item, items, IM_ARRAYSIZE(items)))
    {
        if(current_item == 0)
        {
            Spark::oglContext.resizeWindow(1280, 720);
        }
        else if(current_item == 1)
        {
            Spark::oglContext.resizeWindow(1600, 900);
        }
        else if(current_item == 2)
        {
            Spark::oglContext.resizeWindow(1920, 1080);
        }
        else if(current_item == 3)
        {
            Spark::oglContext.resizeWindow(1920, 1055);
        }
    }

    // TODO: fix this!
    // if (ImGui::Button("Save settings"))
    //{
    //	InitializationVariables variables;
    //	variables.width = Spark::WIDTH;
    //	variables.height = Spark::HEIGHT;
    //	variables.pathToResources = Spark::pathToResources;
    //	variables.pathToModels = Spark::pathToModelMeshes;
    //	JsonSerializer::writeToFile("settings.json", variables.serialize());
    //}
    ImGui::End();
}

int SparkGui::checkCurrentItem(const char** items) const
{
    const std::string resolution = std::to_string(Spark::WIDTH) + "x" + std::to_string(Spark::HEIGHT);
    for(int i = 0; i < 4; i++)
    {
        std::string item(items[i]);
        if(item == resolution)
            return i;
    }
    return 0;
}

std::shared_ptr<Component> SparkGui::addComponent()
{
    std::shared_ptr<Component> component = nullptr;
    if(ImGui::Button("Add Component"))
    {
        ImGui::OpenPopup("Components");
    }

    if(ImGui::BeginPopupModal("Components", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        for(const auto& componentType : componentCreation)
        {
            if(ImGui::Button(componentType.first.c_str()))
            {
                component = componentType.second();
                ImGui::CloseCurrentPopup();
            }
        }
        if(ImGui::Button("Close"))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    return component;
}

std::optional<std::shared_ptr<resources::Model>> SparkGui::getModel()
{
    bool objectPicked{false};
    std::shared_ptr<resources::Model> model{nullptr};
    if(ImGui::Button("Add Model"))
    {
        ImGui::OpenPopup("Models");
    }

    if(ImGui::BeginPopupModal("Models", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        const auto modelIds = Spark::resourceLibrary.getModelResourceIdentifiers();
        for(const auto& id : modelIds)
        {
            if(ImGui::Button(id->getFullPath().string().c_str()))
            {
                model = std::static_pointer_cast<resources::Model>(id->getResource());
                objectPicked = true;
                ImGui::CloseCurrentPopup();
            }
        }
        if(ImGui::Button("Close"))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if(objectPicked)
        return {model};

    return std::nullopt;
}

std::optional<std::shared_ptr<resources::Texture>> SparkGui::getTexture()
{
    bool objectPicked{false};
    std::shared_ptr<resources::Texture> texture{nullptr};
    if(ImGui::Button("Add Texture"))
    {
        ImGui::OpenPopup("Textures");
    }

    if(ImGui::BeginPopupModal("Textures", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        const auto textureIds = Spark::resourceLibrary.getTextureResourceIdentifiers();
        for(const auto& id : textureIds)
        {
            if(ImGui::Button(id->getFullPath().string().c_str()))
            {
                texture = std::static_pointer_cast<resources::Texture>(id->getResource());
                objectPicked = true;
                ImGui::CloseCurrentPopup();
            }
        }

        if(ImGui::Button("Close"))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if(objectPicked)
        return {texture};

    return std::nullopt;
}
}  // namespace spark
