#include "Spark.h"

#include <iostream>

#include <GUI/ImGui/imgui.h>
#include <GUI/ImGui/imgui_impl_glfw.h>
#include <GUI/ImGui/imgui_impl_opengl3.h>

#include "Clock.h"
#include "EngineSystems/SparkRenderer.h"
#include "EngineSystems/SceneManager.h"
#include "HID.h"
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
    resourceLibrary.setup();
    resourceLibrary.createResources(pathToResources);
    SceneManager::getInstance()->setup();

    SparkRenderer::getInstance()->setup();
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
    // glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

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

    glfwSetKeyCallback(window, HID::key_callback);
    glfwSetCursorPosCallback(window, HID::cursor_position_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    setVsync(vsync);

#ifdef DEBUG
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
    glDebugMessageCallback(glDebugOutput, nullptr);
#endif

    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotropicFiltering);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glEnable(GL_CULL_FACE);
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsLight();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    const char* glsl_version = "#version 450";
    ImGui_ImplOpenGL3_Init(glsl_version);
    ImGui_ImplOpenGL3_NewFrame();
}

void Spark::run()
{
    while(!glfwWindowShouldClose(window) && runProgram)
    {
        Clock::tick();
        glfwPollEvents();
        resourceLibrary.processGpuResources();
        SceneManager::getInstance()->update();
        sparkGui.drawGui();
        SparkRenderer::getInstance()->renderPass();
        HID::clearStates();
        // SPARK_DEBUG("FPS: {}", Clock::getFPS());
    }
}

void Spark::resizeWindow(GLuint width, GLuint height)
{
    glfwSetWindowSize(window, width, height);
}

void Spark::setVsync(bool state)
{
    vsync = state;
    if (vsync)
        glfwSwapInterval(1);
    else
        glfwSwapInterval(0);
}

void Spark::clean()
{
    SparkRenderer::getInstance()->cleanup();
    SceneManager::getInstance()->cleanup();
    resourceLibrary.cleanup();
    destroyOpenGLContext();
}

void Spark::destroyOpenGLContext()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}

spark::resourceManagement::ResourceLibrary* Spark::getResourceLibrary()
{
    return &resourceLibrary;
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
