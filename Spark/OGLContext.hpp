#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Logging.h"
#include "HID/HID.h"

namespace spark
{
class OGLContext
{
    public:
    OGLContext() = default;
    ~OGLContext() = default;
    OGLContext(const OGLContext&) = delete;
    OGLContext(OGLContext&&) = delete;
    OGLContext& operator=(const OGLContext&) = delete;
    OGLContext& operator=(OGLContext&&) = delete;

    bool init(unsigned int width, unsigned int height, bool vsyncEnabled, bool isContextOffscreen = false);
    void destroy();

    bool shouldWindowClose() const;
    void closeWindow() const;
    void setVsync(bool vsyncEnabled);
    void resizeWindow(GLuint width, GLuint height) const;
    float maxAnisotropicFiltering() const;
    void setupInputCallbacks() const;

    GLFWwindow* window = nullptr;

    private:
    static void glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
};
}  // namespace spark
