#pragma once

#include <deque>
#include <functional>

#include "glad_glfw3.h"

namespace spark
{
class OpenGLContext
{
    public:
    OpenGLContext(unsigned int width, unsigned int height, bool vsyncEnabled, bool isContextOffscreen = false);
    ~OpenGLContext();
    OpenGLContext(const OpenGLContext&) = delete;
    OpenGLContext(OpenGLContext&&) = delete;
    OpenGLContext& operator=(const OpenGLContext&) = delete;
    OpenGLContext& operator=(OpenGLContext&&) = delete;

    bool shouldWindowClose() const;
    void closeWindow() const;
    void setVsync(bool vsyncEnabled);
    void resizeWindow(GLuint width, GLuint height) const;
    void swapBuffers() const;
    void setupCallbacks() const;
    void addOnWindowSizeChangedCallback(const std::function<void(unsigned int, unsigned int)>& callback);

    GLFWwindow* window{nullptr};

    private:
    bool init(unsigned int width, unsigned int height, bool vsyncEnabled, bool isContextOffscreen);
    void destroy();
    static void windowSizeCallback(GLFWwindow* window, int width, int height);
    static void glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);

    std::deque<std::function<void(unsigned int, unsigned int)>> onSizeChangedCallbacks{};
};
}  // namespace spark
