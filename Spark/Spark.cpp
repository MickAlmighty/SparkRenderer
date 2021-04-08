#include "Spark.h"

#include <GUI/ImGui/imgui.h>
#include <GUI/ImGui/imgui_impl_glfw.h>
#include <GUI/ImGui/imgui_impl_opengl3.h>
#include <stb_image/stb_image.h>

#include "Clock.h"
#include "EngineSystems/SparkRenderer.h"
#include "EngineSystems/SceneManager.h"
#include "HID/HID.h"
#include "JsonSerializer.h"
#include "Logging.h"
#include "SparkConfig.hpp"

namespace spark
{
void Spark::loadConfig(const SparkConfig& config)
{
    WIDTH = config.width;
    HEIGHT = config.height;
    pathToModelMeshes = config.pathToModels;
    pathToResources = config.pathToResources;
    vsync = config.vsync;
}

void Spark::setup()
{
    if(!oglContext.init(WIDTH, HEIGHT, vsync, false))
    {
        SPARK_CRITICAL("oglContext init failed");
        throw std::exception("oglContext init failed");
    }

    oglContext.setupInputCallbacks();

    initImGui();
    createCustomCursor();
    resourceLibrary.setup(pathToResources);
    SceneManager::getInstance()->setup();

    SparkRenderer::getInstance()->setup(WIDTH, HEIGHT);
    SparkRenderer::getInstance()->setScene(SceneManager::getInstance()->getCurrentScene());
}

void Spark::run()
{
    while(!oglContext.shouldWindowClose())
    {
        Clock::tick();
        glfwPollEvents();

        if(HID::isKeyPressed(Key::ESC))
            oglContext.closeWindow();

        int width{}, height{};
        glfwGetWindowSize(oglContext.window, &width, &height);
        if(WIDTH != static_cast<unsigned int>(width) || HEIGHT != static_cast<unsigned int>(height))
        {
            if(width != 0 && height != 0)
            {
                WIDTH = width;
                HEIGHT = height;
            }
        }
        glViewport(0, 0, WIDTH,  HEIGHT);

        SceneManager::getInstance()->update();

        SparkRenderer::getInstance()->renderPass(WIDTH, HEIGHT);
        sparkGui.drawGui();
        oglContext.swapBuffers();

        HID::updateStates();
    }
}

void Spark::clean()
{
    SparkRenderer::getInstance()->cleanup();
    SceneManager::getInstance()->cleanup();
    resourceLibrary.cleanup();
    destroyImGui();
    oglContext.destroy();
}

void Spark::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsLight();

    ImGui_ImplGlfw_InitForOpenGL(oglContext.window, true);
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

void Spark::createCustomCursor()
{
    int width{0};
    int height{0};
    int channels{0};
    unsigned char* pixels = stbi_load((pathToResources / "cursor.png").string().c_str(), &width, &height, &channels, 4);

    GLFWimage image;
    image.width = width;
    image.height = height;
    image.pixels = pixels;

    GLFWcursor* cursor = glfwCreateCursor(&image, 0, 0);

    glfwSetCursor(oglContext.window, cursor);
    ImGui_implGlfw_SetMouseCursor(ImGuiMouseCursor_Arrow, cursor);

    stbi_image_free(pixels);
}

}  // namespace spark
