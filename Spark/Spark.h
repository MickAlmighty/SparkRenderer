#pragma once

#include "GUI/SparkGui.h"
#include "OpenGLContext.hpp"
#include "ResourceLibrary.h"
#include "EngineSystems/SparkRenderer.h"

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

    static Spark& get();
    static void run(const SparkConfig& config);

    virtual OpenGLContext& getRenderingContext() const;
    virtual resourceManagement::ResourceLibrary& getResourceLibrary() const;
    virtual SparkRenderer& getRenderer() const;
    virtual SceneManager& getSceneManager() const;

    unsigned int WIDTH{1280};
    unsigned int HEIGHT{720};
    std::filesystem::path pathToResources{};
    bool vsync = true;

    protected:
    Spark() = default;
    virtual ~Spark() = default;

    private:
    void loadConfig(const SparkConfig& config);
    void setup();
    void runLoop();
    void destroy();

    void initImGui();
    void destroyImGui();

    std::unique_ptr<OpenGLContext> renderingContext{};
    std::unique_ptr<resourceManagement::ResourceLibrary> resourceLibrary{};
    std::unique_ptr<SparkRenderer> renderer{};
    std::unique_ptr<SceneManager> sceneManager{};
    SparkGui sparkGui{};

    inline static Spark* ptr{nullptr};

    friend class SparkInstanceInjector;
};
}  // namespace spark