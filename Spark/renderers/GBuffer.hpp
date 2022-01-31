#pragma once

#include <functional>
#include <memory>

#include "Buffer.hpp"
#include "Enums.h"
#include "glad_glfw3.h"
#include "RenderingRequest.h"
#include "utils/GlHandle.hpp"

namespace spark::resources
{
class Shader;
}

namespace spark::renderers
{
class GBuffer
{
    public:
    GBuffer(unsigned int width, unsigned int height);
    GBuffer(const GBuffer& gBuffer) = delete;
    GBuffer(const GBuffer&& gBuffer) = delete;
    GBuffer operator=(const GBuffer& gBuffer) const = delete;
    GBuffer operator=(const GBuffer&& gBuffer) const = delete;
    ~GBuffer();

    void resize(unsigned int width, unsigned int height);

    void fill(const std::map<ShaderType, std::deque<RenderingRequest>>& renderingQueues, const UniformBuffer& cameraUbo);
    void fill(const std::map<ShaderType, std::deque<RenderingRequest>>& renderingQueues,
              const std::function<bool(const RenderingRequest& request)>& filter, const UniformBuffer& cameraUbo);

    GLuint framebuffer{};
    utils::TextureHandle colorTexture{}, normalsTexture{}, roughnessMetalnessTexture{}, depthTexture{};

    private:
    void createFrameBuffersAndTextures();

    unsigned int w{}, h{};
    std::shared_ptr<resources::Shader> pbrGeometryBufferShader{nullptr};
};
}  // namespace spark::renderers
