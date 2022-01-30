#pragma once

#include "Cube.hpp"
#include "GBuffer.hpp"
#include "glad_glfw3.h"
#include "effects/SkyboxPass.hpp"
#include "utils/GlHandle.hpp"

namespace spark::lights
{
class LightProbe;
}

namespace spark::renderers
{
class LightProbesRenderer
{
    public:
    LightProbesRenderer();
    LightProbesRenderer(const LightProbesRenderer&) = delete;
    LightProbesRenderer(LightProbesRenderer&&) = delete;
    LightProbesRenderer& operator=(const LightProbesRenderer&) = delete;
    LightProbesRenderer& operator=(LightProbesRenderer&&) = delete;
    ~LightProbesRenderer() = default;

    void process(const std::shared_ptr<Scene>& scene);

    private:
    bool checkIfSkyboxChanged(const std::shared_ptr<PbrCubemapTexture>& cubemap) const;
    void generateLightProbe(const std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, lights::LightProbe* lightProbe,
                            const std::shared_ptr<PbrCubemapTexture>& cubemap);
    void renderSceneToCubemap(const std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue,
                              const std::shared_ptr<PbrCubemapTexture>& cubemap);

    const unsigned int sceneCubemapSize{256};
    utils::TextureHandle lightProbeSceneCubemap{};
    GLuint lightProbeLightFbo{};
    GLuint lightProbeSkyboxFbo{};

    ScreenQuad screenQuad{};
    Cube cube{};
    GBuffer localLightProbeGBuffer;
    SSBO cubemapViewMatrices{};
    UniformBuffer cameraUbo{};
    effects::SkyboxPass skyboxPass;

    std::shared_ptr<resources::Shader> localLightProbesLightingShader{nullptr};
    std::shared_ptr<resources::Shader> equirectangularToCubemapShader{nullptr};
    std::shared_ptr<resources::Shader> irradianceShader{nullptr};
    std::shared_ptr<resources::Shader> prefilterShader{nullptr};
    std::shared_ptr<resources::Shader> resampleCubemapShader{nullptr};
};
}  // namespace spark::renderers