#include "Spark.h"

#include "GUI/ImGui/imgui.h"
#include "GUI/ImGui/imgui_impl_glfw.h"
#include "GUI/ImGui/imgui_impl_opengl3.h"

#include "Camera.hpp"
#include "CameraManager.hpp"
#include "Clock.h"
#include "EditorCamera.hpp"
#include "HID/HID.h"
#include "JsonSerializer.h"
#include "Logging.h"
#include "OpenGLContext.hpp"
#include "SparkConfig.hpp"
#include "renderers/RendererFactory.hpp"

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
        if(ImGui::BeginMenu("Rendering Algorithms"))
        {
            using RendererType = renderers::RendererType;
            const unsigned int width = renderingContext->width, height = renderingContext->height;

            constexpr std::array<const std::pair<const char*, const RendererType>, 8> radioButtonsData{
                std::make_pair("Deferred", RendererType::DEFERRED),
                std::make_pair("Forward Plus", RendererType::FORWARD_PLUS),
                std::make_pair("Tile Based Deferred", RendererType::TILE_BASED_DEFERRED),
                std::make_pair("Tile Based Forward Plus", RendererType::TILE_BASED_FORWARD_PLUS),
                std::make_pair("Cluster Based Deferred", RendererType::CLUSTER_BASED_DEFERRED),
                std::make_pair("Cluster Based Forward Plus", RendererType::CLUSTER_BASED_FORWARD_PLUS),
                std::make_pair("Enhanced Cluster Based Deferred", RendererType::ENHANCED_CLUSTER_BASED_DEFERRED),
                std::make_pair("Enhanced Cluster Based Forward Plus", RendererType::ENHANCED_CLUSTER_BASED_FORWARD_PLUS)};

            for(const auto& [radioButtonName, type] : radioButtonsData)
            {
                if(ImGui::RadioButton(radioButtonName, rendererType == type))
                {
                    selectRenderer(type, width, height);
                }
            }

            ImGui::EndMenu();
        }

        renderer->drawGui();
        ImGui::EndMenu();
    }
}

void Spark::loadConfig(const SparkConfig& config)
{
    renderingContext = std::make_unique<OpenGLContext>(config.width, config.height, config.vsync, false);
    pathToResources = config.pathToResources;
    vsync = config.vsync;
}

void Spark::setup()
{
    findResourceDirectoryPath();
    renderingContext->setupCallbacks();
    resourceLibrary = std::make_unique<resourceManagement::ResourceLibrary>(pathToResources);
    SparkGui::setFilePickerPath(pathToResources.string());
    initImGui();
    sceneManager = std::make_unique<SceneManager>();
    selectRenderer(rendererType, renderingContext->width, renderingContext->height);
    animationCreator = std::make_unique<AnimationCreator>();

    renderingContext->addOnWindowSizeChangedCallback([this](auto width, auto height) { renderer->resize(width, height); });
}

void Spark::processKeys()
{
    if(HID::isKeyPressed(Key::ESC))
    {
        renderingContext->closeWindow();
    }
    if(HID::isKeyPressed(Key::F1))
    {
        isEditorEnabled = !isEditorEnabled;
        if(isEditorEnabled)
        {
            glfwSetInputMode(renderingContext->window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        else
        {
            glfwSetInputMode(renderingContext->window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
    }
}

void Spark::runLoop()
{
    while(!renderingContext->shouldWindowClose())
    {
        Clock::tick();
        glfwPollEvents();
        processKeys();
        resourceLibrary->cleanResourceRegistry();

        sceneManager->update();
        if(isEditorEnabled)
        {
            const auto scene = sceneManager->getCurrentScene();
            renderer->render(scene, scene->editorCamera, renderingContext->width, renderingContext->height);
            sparkGui.drawGui();
        }
        else
        {
            const auto scene = sceneManager->getCurrentScene();
            renderer->render(scene, scene->getCameraManager()->getMainCamera(), renderingContext->width, renderingContext->height);
        }

        renderingContext->swapBuffers();

        HID::updateStates();
    }
}

void Spark::destroy()
{
    animationCreator.reset();
    sceneManager.reset();
    renderer.reset();
    resourceLibrary.reset();
    destroyImGui();
    renderingContext.reset();
    delete ptr;
}

void Spark::findResourceDirectoryPath()
{
    if(!std::filesystem::exists(pathToResources))
    {
        constexpr auto resourceDirectoryName = "sparkData";
        SPARK_WARN("Resources directory: {} not found! Searching for default: {}", pathToResources.string(), resourceDirectoryName);

        auto currentDir = std::filesystem::current_path();

        constexpr unsigned int maxDepth = 5;
        for(unsigned int i = 0; i < maxDepth; ++i)
        {
            for(const auto& entry : std::filesystem::directory_iterator(currentDir))
            {
                if(entry.is_directory() && entry.path().filename() == resourceDirectoryName)
                {
                    pathToResources = entry.path();
                    return;
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

AnimationCreator& Spark::getAnimationCreator() const
{
    return *animationCreator;
}

renderers::Renderer& Spark::getRenderer() const
{
    return *renderer;
}

void Spark::selectRenderer(renderers::RendererType type, unsigned int width, unsigned int height)
{
    if(renderer)
    {
        if(type == rendererType && width == renderer->getWidth() && height == renderer->getHeight())
        {
            return;
        }
    }

    rendererType = type;
    renderer = renderers::RendererFactory::createRenderer(type, width, height);
}

renderers::RendererType Spark::getRendererType() const
{
    return rendererType;
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
