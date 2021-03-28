#pragma once

#include "GUI/SparkGui.h"

#include <filesystem>
#include <GLFW/glfw3.h>

namespace spark
{
struct SparkConfig;

namespace resourceManagement
{
    class ResourceLibrary;
}
class Spark
{
    public:
    inline static unsigned int WIDTH{1280};
    inline static unsigned int HEIGHT{720};
    inline static std::filesystem::path pathToModelMeshes;
    inline static std::filesystem::path pathToResources;
    inline static bool vsync = true;
    inline static GLFWwindow* window = nullptr;
    inline static bool runProgram = true;
    inline static float maxAnisotropicFiltering = 1.0f;

    static void loadConfig(const SparkConfig& config);
    static void setup();
    static void run();
    static void resizeWindow(GLuint width, GLuint height);
    static void setVsync(bool state);
    static void clean();

    static void initOpenGL();
    static void destroyOpenGLContext();
    static spark::resourceManagement::ResourceLibrary* getResourceLibrary();

    private:
    inline static SparkGui sparkGui{};
    static spark::resourceManagement::ResourceLibrary resourceLibrary;  // initialized in Spark.cpp

    ~Spark() = default;
    Spark() = default;

    static void initImGui();
    static void destroyImGui();

    static void createCustomCursor();
};
}  // namespace spark