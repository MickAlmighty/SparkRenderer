#pragma once
#include "Component.h"
#include "renderers/RendererType.hpp"

namespace spark::scripts
{
class RendererBenchmark : public Component
{
    public:
    void update() override;

    protected:
    void drawUIBody() override;

    private:
    void startBenchmark();
    void stopBenchmark();
    void generateGameObjectsWithLights();
    void releaseLights() const;


    bool benchmarkStarted{false};
    int numberOfLights{0};
    std::weak_ptr<GameObject> lightContainer;

    renderers::RendererType selectedRenderer{};

    inline const static std::map<const renderers::RendererType, const char*> radioButtonsData{
        std::make_pair(renderers::RendererType::DEFERRED, "Deferred"),
        std::make_pair(renderers::RendererType::FORWARD_PLUS, "Forward Plus"),
        std::make_pair(renderers::RendererType::TILE_BASED_DEFERRED, "Tile Based Deferred"),
        std::make_pair(renderers::RendererType::TILE_BASED_FORWARD_PLUS, "Tile Based Forward Plus"),
        std::make_pair(renderers::RendererType::CLUSTER_BASED_DEFERRED, "Cluster Based Deferred"),
        std::make_pair(renderers::RendererType::CLUSTER_BASED_FORWARD_PLUS, "Cluster Based Forward Plus")};

    RTTR_ENABLE(Component)
};
}  // namespace spark::scripts