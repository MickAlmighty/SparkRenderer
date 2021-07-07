#pragma once

#include <glad/glad.h>

#include "Enums.h"
#include "GBuffer.h"
#include "Scene.h"
#include "Structs.h"
#include "RenderingRequest.h"

namespace spark
{
class BlurPass;
class DepthOfFieldPass;
class LightProbe;
class Shader;
struct PbrCubemapTexture;

class SparkRenderer
{
    public:
    SparkRenderer(const SparkRenderer&) = delete;
    SparkRenderer operator=(const SparkRenderer&) = delete;
    static SparkRenderer* getInstance();

    void setup(unsigned int windowWidth, unsigned int windowHeight);
    void renderPass(unsigned int windowWidth, unsigned int windowHeight);
    void cleanup();

    void drawGui();
    void addRenderingRequest(const RenderingRequest& request);
    void setScene(const std::shared_ptr<Scene>& scene_);
    void setCubemap(const std::shared_ptr<PbrCubemapTexture>& cubemap);

    private:
    SparkRenderer() = default;
    ~SparkRenderer() = default;

    void updateBufferBindings() const;
    void updateLightBuffersBindings() const;
    void resizeWindowIfNecessary(unsigned int windowWidth, unsigned int windowHeight);
    void fillGBuffer(const GBuffer& geometryBuffer);
    void fillGBuffer(const GBuffer& geometryBuffer, const std::function<bool(const RenderingRequest& request)>& filter);
    void ssaoComputing(const GBuffer& geometryBuffer);
    void renderLights(GLuint framebuffer, const GBuffer& geometryBuffer);
    void renderCubemap(GLuint framebuffer) const;
    void tileBasedLightCulling(const GBuffer& geometryBuffer) const;
    void tileBasedLightRendering(const GBuffer& geometryBuffer);
    void bloom();
    void lightShafts();
    void helperShapes();
    void depthOfField();
    void fxaa();
    void motionBlur();
    void toneMapping();
    void calculateAverageLuminance();
    void renderToScreen() const;
    void clearRenderQueues();
    void initMembers();
    void createFrameBuffersAndTextures();
    void deleteFrameBuffersAndTextures();

    static void enableWireframeMode();
    static void disableWireframeMode();

    void updateCameraUBO(glm::mat4 projection, glm::mat4 view, glm::vec3 pos);
    bool checkIfSkyboxChanged() const;
    void lightProbesRenderPass();
    void generateLightProbe(LightProbe* lightProbe);
    void renderSceneToCubemap(const GBuffer& geometryBuffer, GLuint lightFbo, GLuint skyboxFbo);

    std::map<ShaderType, std::deque<RenderingRequest>> renderQueue{};
    std::shared_ptr<Scene> scene{nullptr};

    unsigned int width{}, height{};

    std::weak_ptr<PbrCubemapTexture> pbrCubemap;

    ScreenQuad screenQuad{};
    std::unique_ptr<DepthOfFieldPass> dofPass;
    std::unique_ptr<BlurPass> ssaoBlurPass;

    GBuffer gBuffer{};

    GLuint lightFrameBuffer{}, lightColorTexture{}, brightPassTexture{};
    GLuint cubemapFramebuffer{};
    GLuint toneMappingFramebuffer{}, toneMappingTexture{};
    GLuint motionBlurFramebuffer{}, motionBlurTexture{};
    GLuint lightShaftFramebuffer{}, lightShaftTexture{};
    GLuint fxaaFramebuffer{}, fxaaTexture{};
    GLuint averageLuminanceTexture{};
    GLuint lightsPerTileTexture{};

    // IBL
    GLuint brdfLookupTexture{};

    // bloom start
    std::unique_ptr<BlurPass> upsampleBloomBlurPass2;
    std::unique_ptr<BlurPass> upsampleBloomBlurPass4;
    std::unique_ptr<BlurPass> upsampleBloomBlurPass8;
    std::unique_ptr<BlurPass> upsampleBloomBlurPass16;

    GLuint bloomFramebuffer{}, bloomTexture{};
    GLuint downsampleFramebuffer2{}, downsampleTexture2{};
    GLuint downsampleFramebuffer4{}, downsampleTexture4{};
    GLuint downsampleFramebuffer8{}, downsampleTexture8{};
    GLuint downsampleFramebuffer16{}, downsampleTexture16{};
    // bloom end

    // local light probes start
    const unsigned int sceneCubemapSize{256};
    GLuint lightProbeSceneCubemap{};
    GLuint lightProbeLightFbo{};
    GLuint lightProbeSkyboxFbo{};
    GBuffer localLightProbeGBuffer{};
    SSBO cubemapViewMatrices{};
    // local light probes end

    GLuint ssaoFramebuffer{}, ssaoTexture{}, randomNormalsTexture{}, ssaoDisabledTexture{};

    GLuint textureHandle{};  // temporary, its only a handle to other texture -> dont delete it

    std::shared_ptr<resources::Shader> mainShader{nullptr};
    std::shared_ptr<resources::Shader> screenShader{nullptr};
    std::shared_ptr<resources::Shader> toneMappingShader{nullptr};
    std::shared_ptr<resources::Shader> lightShader{nullptr};
    std::shared_ptr<resources::Shader> motionBlurShader{nullptr};
    std::shared_ptr<resources::Shader> cubemapShader{nullptr};
    std::shared_ptr<resources::Shader> ssaoShader{nullptr};
    std::shared_ptr<resources::Shader> circleOfConfusionShader{nullptr};
    std::shared_ptr<resources::Shader> bokehDetectionShader{nullptr};
    std::shared_ptr<resources::Shader> blendDofShader{nullptr};
    std::shared_ptr<resources::Shader> solidColorShader{nullptr};
    std::shared_ptr<resources::Shader> lightShaftsShader{nullptr};
    std::shared_ptr<resources::Shader> luminanceHistogramComputeShader{nullptr};
    std::shared_ptr<resources::Shader> averageLuminanceComputeShader{nullptr};
    std::shared_ptr<resources::Shader> fxaaShader{nullptr};
    std::shared_ptr<resources::Shader> bloomDownScaleShader{nullptr};
    std::shared_ptr<resources::Shader> bloomUpScaleShader{nullptr};
    std::shared_ptr<resources::Shader> tileBasedLightCullingShader{nullptr};
    std::shared_ptr<resources::Shader> tileBasedLightingShader{nullptr};
    std::shared_ptr<resources::Shader> localLightProbesLightingShader{nullptr};
    std::shared_ptr<resources::Shader> equirectangularToCubemapShader{nullptr};
    std::shared_ptr<resources::Shader> irradianceShader{nullptr};
    std::shared_ptr<resources::Shader> prefilterShader{nullptr};
    std::shared_ptr<resources::Shader> resampleCubemapShader{nullptr};

    Cube cube = Cube();
    UniformBuffer cameraUBO{};
    UniformBuffer sampleUniformBuffer{};
    SSBO luminanceHistogram{};
    SSBO pointLightIndices{};
    SSBO spotLightIndices{};
    SSBO lightProbeIndices{};

    bool ssaoEnable = false;
    int kernelSize = 32;
    float radius = 0.7f;
    float bias = 0.035f;
    float power = 4.0f;

    bool dofEnable = false;
    float nearStart = 1.0f;
    float nearEnd = 4.0f;
    float farStart = 20.0f;
    float farEnd = 100.0f;

    bool lightShaftsEnable = false;
    int samples = 100;
    float exposure = 0.0034f;
    float decay = 0.995f;
    float density = 0.75f;
    float weight = 6.65f;

    // tone mapping
    float minLogLuminance = 0.5f;
    float oneOverLogLuminanceRange = 1.0f / 12.0f;
    float logLuminanceRange = 12.0f;
    float tau = 1.1f;

    bool bloomEnable = false;
    float bloomIntensity = 0.1f;

    bool motionBlurEnable = true;
};
}  // namespace spark