#pragma once

#include "Buffer.hpp"
#include "RenderingRequest.h"
#include "effects/AmbientOcclusion.hpp"

namespace spark::renderers
{
class Renderer
{
    public:
    Renderer(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo);
    Renderer(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer& operator=(Renderer&&) = delete;
    virtual ~Renderer() = default;

    virtual GLuint process(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap,
                           const UniformBuffer& cameraUbo) = 0;
    virtual void bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager) = 0;
    virtual void resize(unsigned int width, unsigned int height) = 0;
    virtual GLuint getDepthTexture() const = 0;

    bool isAmbientOcclusionEnabled{false};
    effects::AmbientOcclusion ao;

    protected:
    unsigned int w{}, h{};
};

inline Renderer::Renderer(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo) : ao(width, height, cameraUbo), w(width), h(height)
{
}
}  // namespace spark::renderers