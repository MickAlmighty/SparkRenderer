#pragma once

namespace spark::renderers
{
    enum class RendererType
    {
        FORWARD_PLUS,
        TILE_BASED_FORWARD_PLUS,
        CLUSTER_BASED_FORWARD_PLUS,
        ENHANCED_CLUSTER_BASED_FORWARD_PLUS,
        DEFERRED,
        TILE_BASED_DEFERRED,
        CLUSTER_BASED_DEFERRED,
        ENHANCED_CLUSTER_BASED_DEFERRED
    };
}