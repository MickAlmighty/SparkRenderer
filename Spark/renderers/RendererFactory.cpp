#include "RendererFactory.hpp"

#include "ClusterBasedDeferredRenderer.hpp"
#include "ClusterBasedForwardPlusRenderer.hpp"
#include "DeferredRenderer.hpp"
#include "ForwardPlusRenderer.hpp"
#include "Spark.h"
#include "TileBasedDeferredRenderer.hpp"
#include "TileBasedForwardPlusRenderer.hpp"

namespace
{
using namespace spark;

std::pair<std::shared_ptr<resources::Shader>, std::shared_ptr<resources::Shader>> getBasicShaders()
{
    return {Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/determineActiveCluster.glsl"),
            Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/clusterBasedLightCulling.glsl")};
}

std::pair<std::shared_ptr<resources::Shader>, std::shared_ptr<resources::Shader>> getEnhancedShaders()
{
    return {Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/determineActiveClusterEnhanced.glsl"),
            Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/clusterBasedLightCullingEnhanced.glsl")};
}

std::unique_ptr<renderers::Renderer> createClusterBasedForwardPlusRenderer(unsigned int width, unsigned int height)
{
    const auto [determineActiveClustersShader, clusterBasedLightCullingShader] = getBasicShaders();
    return std::make_unique<renderers::ClusterBasedForwardPlusRenderer>(width, height, determineActiveClustersShader, clusterBasedLightCullingShader);
}

std::unique_ptr<renderers::Renderer> createClusterBasedDeferredRenderer(unsigned int width, unsigned int height)
{
    const auto [determineActiveClustersShader, clusterBasedLightCullingShader] = getBasicShaders();
    return std::make_unique<renderers::ClusterBasedDeferredRenderer>(width, height, determineActiveClustersShader, clusterBasedLightCullingShader);
}

std::unique_ptr<renderers::Renderer> createEnhancedClusterBasedForwardPlusRenderer(unsigned int width, unsigned int height)
{
    const auto [determineActiveClustersShader, clusterBasedLightCullingShader] = getEnhancedShaders();
    return std::make_unique<renderers::ClusterBasedForwardPlusRenderer>(width, height, determineActiveClustersShader, clusterBasedLightCullingShader);
}

std::unique_ptr<renderers::Renderer> createEnhancedClusterBasedDeferredRenderer(unsigned int width, unsigned int height)
{
    const auto [determineActiveClustersShader, clusterBasedLightCullingShader] = getEnhancedShaders();
    return std::make_unique<renderers::ClusterBasedDeferredRenderer>(width, height, determineActiveClustersShader, clusterBasedLightCullingShader);
}
}  // namespace

namespace spark::renderers
{
std::unique_ptr<Renderer> RendererFactory::createRenderer(RendererType type, unsigned int width, unsigned int height)
{
    switch(type)
    {
        case RendererType::FORWARD_PLUS:
            return std::make_unique<ForwardPlusRenderer>(width, height);
        case RendererType::TILE_BASED_FORWARD_PLUS:
            return std::make_unique<TileBasedForwardPlusRenderer>(width, height);
        case RendererType::CLUSTER_BASED_FORWARD_PLUS:
            return createClusterBasedForwardPlusRenderer(width, height);
        case RendererType::DEFERRED:
            return std::make_unique<DeferredRenderer>(width, height);
        case RendererType::TILE_BASED_DEFERRED:
            return std::make_unique<TileBasedDeferredRenderer>(width, height);
        case RendererType::ENHANCED_CLUSTER_BASED_FORWARD_PLUS:
            return createEnhancedClusterBasedForwardPlusRenderer(width, height);
        case RendererType::ENHANCED_CLUSTER_BASED_DEFERRED:
            return createEnhancedClusterBasedDeferredRenderer(width, height);
        case RendererType::CLUSTER_BASED_DEFERRED:
        default:
            return createClusterBasedDeferredRenderer(width, height);
    }
}
}  // namespace spark::renderers
