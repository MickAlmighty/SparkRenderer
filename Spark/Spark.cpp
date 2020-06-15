#include "Spark.h"

#include <iostream>

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
#include "ResourceLibrary.h"

namespace spark
{
spark::resourceManagement::ResourceLibrary Spark::resourceLibrary = resourceManagement::ResourceLibrary();

void Spark::setInitVariables(const InitializationVariables& variables)
{
    WIDTH = variables.width;
    HEIGHT = variables.height;
    pathToModelMeshes = variables.pathToModels;
    pathToResources = variables.pathToResources;
    vsync = variables.vsync;
}

void Spark::setup()
{
    initOpenGL();
    initImGui();
    createCustomCursor();
    resourceLibrary.setup();
    resourceLibrary.createResources(pathToResources);
    SceneManager::getInstance()->setup();

    SparkRenderer::getInstance()->setup();
}

void Spark::run()
{
    while(!glfwWindowShouldClose(window) && runProgram)
    {
        Clock::tick();
        glfwPollEvents();

        static bool mouseEnabled = true;

        if(HID::isKeyReleased(Key::LEFT_ALT) && !mouseEnabled)
        {
            glfwSetInputMode(Spark::window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            mouseEnabled = true;
        }
        else if(HID::isKeyReleased(Key::LEFT_ALT) && mouseEnabled)
        {
            glfwSetInputMode(Spark::window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            mouseEnabled = false;
        }

        if(HID::isKeyPressed(Key::ESC))
            glfwSetWindowShouldClose(Spark::window, GLFW_TRUE);

        resourceLibrary.processGpuResources();
        SceneManager::getInstance()->update();
        sparkGui.drawGui();
        SparkRenderer::getInstance()->renderPass();
        HID::processInputDevicesStates();
    }
}

void Spark::clean()
{
    SparkRenderer::getInstance()->cleanup();
    SceneManager::getInstance()->cleanup();
    resourceLibrary.cleanup();
    destroyImGui();
    destroyOpenGLContext();
}

void APIENTRY glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);

void Spark::initOpenGL()
{
    if(!glfwInit())
    {
        SPARK_CRITICAL("glfw init failed");
        throw std::exception("glfw init failed");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
#ifdef DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#else
    glfwWindowHint(GLFW_CONTEXT_NO_ERROR, GL_TRUE);
    SPARK_INFO("Created context with disabled gl errors checking and reporting.");
#endif

    window = glfwCreateWindow(static_cast<int>(Spark::WIDTH), static_cast<int>(Spark::HEIGHT), "Spark", nullptr, nullptr);
    if(!window)
    {
        SPARK_CRITICAL("Window creation failed");
        throw std::exception("Window creation failed");
    }

    glfwMakeContextCurrent(window);

    if(!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        SPARK_CRITICAL("Failed to initialize OpenGL loader!");
        throw std::exception("Failed to initialize OpenGL loader!");
    }

#ifdef DEBUG
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(glDebugOutput, nullptr);
#else

#endif

    glfwSetKeyCallback(window, HID::key_callback);
    glfwSetCursorPosCallback(window, HID::cursor_position_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    setVsync(vsync);

    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAnisotropicFiltering);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glEnable(GL_CULL_FACE);
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);

    const GLubyte* const vendorName = glGetString(GL_VENDOR);
    const GLubyte* const deviceName = glGetString(GL_RENDERER);

    SPARK_INFO("OpenGL Context initialized!");
    SPARK_INFO("Device: {} {}", vendorName, deviceName);
}

void Spark::resizeWindow(GLuint width, GLuint height)
{
    glfwSetWindowSize(window, width, height);
}

void Spark::setVsync(bool state)
{
    vsync = state;
    if(vsync)
        glfwSwapInterval(1);
    else
        glfwSwapInterval(0);
}

void Spark::destroyOpenGLContext()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

spark::resourceManagement::ResourceLibrary* Spark::getResourceLibrary()
{
    return &resourceLibrary;
}

void Spark::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsLight();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
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

    glfwSetCursor(window, cursor);
    ImGui_implGlfw_SetMouseCursor(ImGuiMouseCursor_Arrow, cursor);

    stbi_image_free(pixels);
}

void APIENTRY glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{
    // ignore non-significant error/warning codes
    if(id == 131169 || id == 131185 || id == 131218 || id == 131204)
        return;

    std::string msg{"[GL{}] "};

    switch(source)
    {
        case GL_DEBUG_SOURCE_API:
            msg.append("[API] ");
            break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
            msg.append("[WinSys] ");
            break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER:
            msg.append("[ShadComp] ");
            break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:
            msg.append("[3rdParty] ");
            break;
        case GL_DEBUG_SOURCE_APPLICATION:
            msg.append("[App] ");
            break;
        case GL_DEBUG_SOURCE_OTHER:
            msg.append("[Other] ");
            break;
    }

    switch(severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:
            msg.append("[H] ");
            break;
        case GL_DEBUG_SEVERITY_MEDIUM:
            msg.append("[M] ");
            break;
        case GL_DEBUG_SEVERITY_LOW:
            msg.append("[L] ");
            break;
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            msg.append("[N] ");
            break;
    }

    switch(type)
    {
        case GL_DEBUG_TYPE_ERROR:
            msg.append("[Err] {}");
            SPARK_ERROR(msg, id, message);
            break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
            msg.append("[DepBeh] {}");
            SPARK_WARN(msg, id, message);
            break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
            msg.append("[UndefBeh] {}");
            SPARK_WARN(msg, id, message);
            break;
        case GL_DEBUG_TYPE_PORTABILITY:
            msg.append("[Port] {}");
            SPARK_DEBUG(msg, id, message);
            break;
        case GL_DEBUG_TYPE_PERFORMANCE:
            msg.append("[Perf] {}");
            SPARK_DEBUG(msg, id, message);
            break;
#ifdef SPARK_LOG_GL_MARKERS  // located in Logging.h
        case GL_DEBUG_TYPE_MARKER:
            msg.append("[Marker] {}");
            SPARK_TRACE(msg, id, message);
            break;
        case GL_DEBUG_TYPE_PUSH_GROUP:
            msg.append("[Push] {}");
            SPARK_TRACE(msg, id, message);
            break;
        case GL_DEBUG_TYPE_POP_GROUP:
            msg.append("[Pop] {}");
            SPARK_TRACE(msg, id, message);
            break;
#endif
        case GL_DEBUG_TYPE_OTHER:
            msg.append("[Other] {}");
            SPARK_DEBUG(msg, id, message);
            break;
    }
}

}  // namespace spark
