#pragma once

#include <memory>

#include "Renderer.hpp"
#include "RendererType.hpp"

namespace spark::renderers
{
class RendererFactory
{
    public:
    [[nodiscard]] static std::unique_ptr<Renderer> createRenderer(RendererType type, unsigned int width, unsigned int height);

    RendererFactory() = delete;
    RendererFactory(const RendererFactory&) = delete;
    RendererFactory(RendererFactory&&) = delete;
    RendererFactory& operator=(const RendererFactory&) = delete; 
    RendererFactory& operator=(RendererFactory&&) = delete; 
    ~RendererFactory() = delete;
};
}