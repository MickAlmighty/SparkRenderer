#pragma once

#include "GUI/SparkGui.h"
#include "OGLContext.hpp"

#include <filesystem>

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
    Spark(const Spark&) = delete;
    Spark(Spark&&) = delete;
    Spark& operator=(const Spark&) = delete;
    Spark& operator=(Spark&&) = delete;

    inline static unsigned int WIDTH{1280};
    inline static unsigned int HEIGHT{720};
    inline static std::filesystem::path pathToModelMeshes{};
    inline static std::filesystem::path pathToResources{};
    inline static bool vsync = true;
    inline static bool runProgram = true;

    inline static OGLContext oglContext{};

    static void loadConfig(const SparkConfig& config);
    static void setup();
    static void run();
    static void clean();

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