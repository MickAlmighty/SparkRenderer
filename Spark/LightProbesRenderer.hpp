#pragma once

#include "Cube.hpp"
#include "GBuffer.h"
#include "glad_glfw3.h"
#include "effects/SkyboxPass.hpp"

namespace spark
{
class LightProbesRenderer
{
    public:
    LightProbesRenderer() = default;
    LightProbesRenderer(const LightProbesRenderer&) = delete;
    LightProbesRenderer(LightProbesRenderer&&) = delete;
    LightProbesRenderer& operator=(const LightProbesRenderer&) = delete;
    LightProbesRenderer& operator=(LightProbesRenderer&&) = delete;
    ~LightProbesRenderer() = default;

    void setup(const std::shared_ptr<lights::LightManager>& lightManager);
    void process(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const std::shared_ptr<PbrCubemapTexture>& cubemap,
                 const std::vector<lights::LightProbe*>& lightProbes);
    void bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager);

    private:
    bool checkIfSkyboxChanged(const std::shared_ptr<PbrCubemapTexture>& cubemap) const;
    void generateLightProbe(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, lights::LightProbe* lightProbe,
                            const std::shared_ptr<PbrCubemapTexture>& cubemap);
    void renderSceneToCubemap(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const std::shared_ptr<PbrCubemapTexture>& cubemap);

    const unsigned int sceneCubemapSize{256};
    GLuint lightProbeSceneCubemap{};
    GLuint lightProbeLightFbo{};
    GLuint lightProbeSkyboxFbo{};

    ScreenQuad screenQuad{};
    Cube cube{};
    GBuffer localLightProbeGBuffer{};
    SSBO cubemapViewMatrices{};
    UniformBuffer cameraUbo{};
    effects::SkyboxPass skyboxPass{};

    std::shared_ptr<resources::Shader> localLightProbesLightingShader{nullptr};
    std::shared_ptr<resources::Shader> equirectangularToCubemapShader{nullptr};
    std::shared_ptr<resources::Shader> irradianceShader{nullptr};
    std::shared_ptr<resources::Shader> prefilterShader{nullptr};
    std::shared_ptr<resources::Shader> resampleCubemapShader{nullptr};
};
}  // namespace spark