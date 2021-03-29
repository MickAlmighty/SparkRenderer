#include "OGLContext.hpp"

namespace spark
{
bool OGLContext::init(unsigned width, unsigned height, bool vsyncEnabled, bool isContextOffscreen)
{
    if(!glfwInit())
    {
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    if(isContextOffscreen)
    {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    }

    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

#ifdef DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#else
    glfwWindowHint(GLFW_CONTEXT_NO_ERROR, GL_TRUE);
    SPARK_INFO("Created context with disabled gl errors checking and reporting.");
#endif

    window = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), "SparkRenderer", nullptr, nullptr);
    if(!window)
    {
        return false;
    }

    glfwMakeContextCurrent(window);

    if(!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        return false;
    }

#ifdef DEBUG
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(glDebugOutput, nullptr);
#else

#endif

    // glfwSetWindowUserPointer(window, this);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    setVsync(vsyncEnabled);

    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glEnable(GL_CULL_FACE);
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);

    const GLubyte* const vendorName = glGetString(GL_VENDOR);
    const GLubyte* const deviceName = glGetString(GL_RENDERER);

    SPARK_INFO("OpenGL Context initialized!");
    SPARK_INFO("Device: {} {}", vendorName, deviceName);

    return true;
}

void OGLContext::destroy()
{
    glfwDestroyWindow(window);
    window = nullptr;
    glfwTerminate();
}

bool OGLContext::shouldWindowClose() const
{
    return glfwWindowShouldClose(window);
}

void OGLContext::closeWindow() const
{
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void OGLContext::setVsync(bool vsyncEnabled)
{
    if(vsyncEnabled)
        glfwSwapInterval(1);
    else
        glfwSwapInterval(0);
}

void OGLContext::resizeWindow(GLuint width, GLuint height) const
{
    glfwSetWindowSize(window, width, height);
}

float OGLContext::maxAnisotropicFiltering() const
{
    float maxAnisotropicFiltering{1.0f};
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAnisotropicFiltering);
    return maxAnisotropicFiltering;
}

void OGLContext::setupInputCallbacks() const
{
    glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int scancode, int action, int mods) {
        HID::key_callback(key, scancode, action, mods);
        // const auto* oglContext = static_cast<OGLContext*>(glfwGetWindowUserPointer(w));
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double xpos, double ypos) { HID::cursor_position_callback(xpos, ypos); });
    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int button, int action, int mods) { HID::mouse_button_callback(button, action, mods); });
    glfwSetScrollCallback(window, [](GLFWwindow* w, double xoffset, double yoffset) { HID::scroll_callback(xoffset, yoffset); });
}

void OGLContext::glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
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
