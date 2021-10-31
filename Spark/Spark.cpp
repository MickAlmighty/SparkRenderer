#include "Spark.h"

#include "GUI/ImGui/imgui.h"
#include "GUI/ImGui/imgui_impl_glfw.h"
#include "GUI/ImGui/imgui_impl_opengl3.h"

#include "Clock.h"
#include "HID/HID.h"
#include "JsonSerializer.h"
#include "Logging.h"
#include "OpenGLContext.hpp"
#include "SparkConfig.hpp"
#include "renderers/ClusterBasedDeferredRenderer.hpp"
#include "renderers/ClusterBasedForwardPlusRenderer.hpp"
#include "renderers/DeferredRenderer.hpp"
#include "renderers/ForwardPlusRenderer.hpp"
#include "renderers/TileBasedDeferredRenderer.hpp"
#include "renderers/TileBasedForwardPlusRenderer.hpp"

namespace spark
{
Spark& Spark::get()
{
    if(!ptr)
    {
        ptr = new Spark();
    }
    return *ptr;
}

void Spark::run(const SparkConfig& config)
{
    auto& engine = get();
    engine.loadConfig(config);
    engine.setup();
    engine.runLoop();
    engine.destroy();
}

void Spark::drawGui()
{
    if(ImGui::BeginMenu("Renderer"))
    {
        using namespace renderers;
        const std::string renderingAlgorithmTypeMenu = "Rendering Algorithms";
        if(ImGui::BeginMenu(renderingAlgorithmTypeMenu.c_str()))
        {
            static int mode = 2;

            if(ImGui::RadioButton("Deferred", mode == 0))
            {
                mode = 0;
                renderer = std::make_unique<DeferredRenderer>(WIDTH, HEIGHT);
            }
            if(ImGui::RadioButton("Forward Plus", mode == 1))
            {
                mode = 1;
                renderer = std::make_unique<ForwardPlusRenderer>(WIDTH, HEIGHT);
            }
            if(ImGui::RadioButton("Tile Based Deferred", mode == 2))
            {
                mode = 2;
                renderer = std::make_unique<TileBasedDeferredRenderer>(WIDTH, HEIGHT);
            }
            if(ImGui::RadioButton("Tile Based Forward Plus", mode == 3))
            {
                mode = 3;
                renderer = std::make_unique<TileBasedForwardPlusRenderer>(WIDTH, HEIGHT);
            }
            if(ImGui::RadioButton("Cluster Based Deferred", mode == 4))
            {
                mode = 4;
                renderer = std::make_unique<ClusterBasedDeferredRenderer>(WIDTH, HEIGHT);
            }
            if(ImGui::RadioButton("Cluster Based Forward Plus", mode == 5))
            {
                mode = 5;
                renderer = std::make_unique<ClusterBasedForwardPlusRenderer>(WIDTH, HEIGHT);
            }
            ImGui::EndMenu();
        }

        renderer->drawGui();
        ImGui::EndMenu();
    }
}

void Spark::loadConfig(const SparkConfig& config)
{
    WIDTH = config.width;
    HEIGHT = config.height;
    pathToResources = config.pathToResources;
    vsync = config.vsync;
}

void Spark::setup()
{
    renderingContext = std::make_unique<OpenGLContext>(WIDTH, HEIGHT, vsync, false);
    renderingContext->setupCallbacks();
    const auto resourcePath = std::filesystem::exists(pathToResources) ? pathToResources : findResourceDirectoryPath();
    resourceLibrary = std::make_unique<resourceManagement::ResourceLibrary>(resourcePath);
    SparkGui::setFilePickerPath(resourcePath.string());
    initImGui();
    sceneManager = std::make_unique<SceneManager>();
    renderer = std::make_unique<renderers::TileBasedDeferredRenderer>(WIDTH, HEIGHT);

    renderingContext->addOnWindowSizeChangedCallback([this](auto width, auto height) {
        WIDTH = width;
        HEIGHT = height;
    });
    renderingContext->addOnWindowSizeChangedCallback([this](auto width, auto height) { renderer->resize(width, height); });
}

void Spark::runLoop()
{
    while(!renderingContext->shouldWindowClose())
    {
        Clock::tick();
        glfwPollEvents();

        if(HID::isKeyPressed(Key::ESC))
            renderingContext->closeWindow();

        sceneManager->update();
        renderer->render(sceneManager->getCurrentScene());
        sparkGui.drawGui();
        renderingContext->swapBuffers();

        HID::updateStates();
    }
}

void Spark::destroy()
{
    sceneManager.reset();
    renderer.reset();
    resourceLibrary.reset();
    destroyImGui();
    renderingContext.reset();
    delete ptr;
}

std::filesystem::path Spark::findResourceDirectoryPath() const
{
    constexpr auto resourceDirectoryName = "sparkData";

    auto currentDir = std::filesystem::current_path();

    const unsigned int maxDepth = 5;
    for(unsigned int i = 0; i < maxDepth; ++i)
    {
        for(const auto& entry : std::filesystem::directory_iterator(currentDir))
        {
            if(entry.is_directory() && entry.path().filename() == resourceDirectoryName)
            {
                return entry.path();
            }
        }

        if(currentDir.has_parent_path())
        {
            currentDir = currentDir.parent_path();
        }
        else
        {
            break;
        }
    }

    constexpr auto errorMessage{R"(Resource directory "sparkData" has not been found!)"};
    SPARK_CRITICAL(errorMessage);
    throw std::runtime_error(errorMessage);
}

OpenGLContext& Spark::getRenderingContext() const
{
    return *renderingContext;
}

resourceManagement::ResourceLibrary& Spark::getResourceLibrary() const
{
    return *resourceLibrary;
}

SceneManager& Spark::getSceneManager() const
{
    return *sceneManager;
}

void Spark::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsLight();

    ImGui_ImplGlfw_InitForOpenGL(renderingContext->window, true);
    const char* glsl_version = "#version 450";
    ImGui_ImplOpenGL3_Init(glsl_version);
    // ImGui::SetMouseCursor(ImGuiMouseCursor_None);
    ImGui_ImplOpenGL3_NewFrame();
}

void Spark::destroyImGui()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}
}  // namespace spark
