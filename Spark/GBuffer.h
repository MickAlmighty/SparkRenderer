#pragma once

#include <functional>
#include <memory>

#include "Enums.h"
#include "glad_glfw3.h"
#include "lights/LightManager.h"
#include "RenderingRequest.h"

namespace spark
{
namespace resources
{
    class Shader;
}

class GBuffer
{
    public:
    void createFrameBuffersAndTextures(unsigned int width, unsigned int height);

    void fill(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const UniformBuffer& cameraUbo);
    void fill(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const std::function<bool(const RenderingRequest& request)>& filter,
              const UniformBuffer& cameraUbo);

    GBuffer();
    GBuffer(const GBuffer& gBuffer) = delete;
    GBuffer(const GBuffer&& gBuffer) = delete;
    GBuffer operator=(const GBuffer& gBuffer) const = delete;
    GBuffer operator=(const GBuffer&& gBuffer) const = delete;
    ~GBuffer();

    GLuint framebuffer{};
    GLuint colorTexture{};
    GLuint normalsTexture{};
    GLuint roughnessMetalnessTexture{};
    GLuint depthTexture{};

    private:
    unsigned int w{}, h{};
    std::shared_ptr<resources::Shader> pbrGeometryBufferShader{nullptr};
};
}  // namespace spark
