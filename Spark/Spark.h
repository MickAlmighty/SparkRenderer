#ifndef SPARK_H
#define SPARK_H

#include "GUI/SparkGui.h"
#include "Structs.h"

#include <filesystem>
#include <GLFW/glfw3.h>

namespace spark
{
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
    inline static GLFWwindow* window = nullptr;
    inline static bool runProgram = true;
    inline static float maxAnisotropicFiltering = 1.0f;

    static void setInitVariables(const InitializationVariables& variables);
    static void setup();
    static void run();
    static void resizeWindow(GLuint width, GLuint height);
    static void clean();

    static void initOpenGL();
    static void destroyOpenGLContext();
    static spark::resourceManagement::ResourceLibrary* getResourceLibrary();

    private:
    inline static SparkGui sparkGui{};
    static spark::resourceManagement::ResourceLibrary resourceLibrary; // initialized in Spark.cpp

    ~Spark() = default;
    Spark() = default;
};
}  // namespace spark
#endif