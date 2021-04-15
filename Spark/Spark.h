#pragma once

#include "GUI/SparkGui.h"
#include "OGLContext.hpp"
#include "ResourceLibrary.h"

namespace spark
{
struct SparkConfig;

class Spark
{
    public:
    Spark(const Spark&) = delete;
    Spark(Spark&&) = delete;
    Spark& operator=(const Spark&) = delete;
    Spark& operator=(Spark&&) = delete;

    inline static unsigned int WIDTH{1280};
    inline static unsigned int HEIGHT{720};
    inline static std::filesystem::path pathToResources{};
    inline static bool vsync = true;

    inline static OGLContext oglContext{};
    inline static resourceManagement::ResourceLibrary resourceLibrary{};

    static void loadConfig(const SparkConfig& config);
    static void setup();
    static void run();
    static void clean();

    private:
    inline static SparkGui sparkGui{};

    ~Spark() = default;
    Spark() = default;

    static void initImGui();
    static void destroyImGui();
};
}  // namespace spark