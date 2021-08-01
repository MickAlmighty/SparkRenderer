#include "Spark.h"

#include "GUI/ImGui/imgui.h"
#include "GUI/ImGui/imgui_impl_glfw.h"
#include "GUI/ImGui/imgui_impl_opengl3.h"

#include "Clock.h"
#include "EngineSystems/SparkRenderer.h"
#include "EngineSystems/SceneManager.h"
#include "HID/HID.h"
#include "JsonSerializer.h"
#include "Logging.h"
#include "OpenGLContext.hpp"
#include "SparkConfig.hpp"

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
    resourceLibrary = std::make_unique<resourceManagement::ResourceLibrary>(pathToResources);
    initImGui();
    sceneManager = std::make_unique<SceneManager>();
    renderer = std::make_unique<SparkRenderer>(WIDTH, HEIGHT, sceneManager->getCurrentScene());

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
        renderer->renderPass();
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

OpenGLContext& Spark::getRenderingContext() const
{
    return *renderingContext;
}

resourceManagement::ResourceLibrary& Spark::getResourceLibrary() const
{
    return *resourceLibrary;
}

SparkRenderer& Spark::getRenderer() const
{
    return *renderer;
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
