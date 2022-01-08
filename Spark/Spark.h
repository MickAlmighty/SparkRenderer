#pragma once

#include "AnimationCreator.hpp"
#include "GUI/SparkGui.h"
#include "OpenGLContext.hpp"
#include "ResourceLibrary.h"
#include "SceneManager.h"

namespace spark
{
class AnimationCreator;

namespace renderers
{
    class Renderer;
}

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
    void drawGui();

    virtual OpenGLContext& getRenderingContext() const;
    virtual resourceManagement::ResourceLibrary& getResourceLibrary() const;
    virtual SceneManager& getSceneManager() const;
    virtual AnimationCreator& getAnimationCreator() const;

    renderers::Renderer& getRenderer() const;

    bool vsync = true;
    bool isEditorEnabled{true};

    protected:
    Spark() = default;
    virtual ~Spark() = default;

    void loadConfig(const SparkConfig& config);
    void setup();
    void runLoop();
    void destroy();

    std::filesystem::path findResourceDirectoryPath() const;

    void initImGui();
    void destroyImGui();

    std::unique_ptr<OpenGLContext> renderingContext{};
    std::unique_ptr<resourceManagement::ResourceLibrary> resourceLibrary{};
    std::unique_ptr<renderers::Renderer> renderer{};
    std::unique_ptr<SceneManager> sceneManager{};
    std::unique_ptr<AnimationCreator> animationCreator{};
    SparkGui sparkGui{};
    std::filesystem::path pathToResources{};

    inline static Spark* ptr{nullptr};

    friend class SparkInstanceInjector;
};
}  // namespace spark