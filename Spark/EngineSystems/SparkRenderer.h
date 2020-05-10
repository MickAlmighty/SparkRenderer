#ifndef SPARK_RENDERER_H
#define SPARK_RENDERER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Enums.h"
#include "GBuffer.h"
#include "Structs.h"
#include "Shader.h"

namespace spark
{
class BlurPass;
class DepthOfFieldPass;
class LightProbe;
struct RenderingRequest;

class SparkRenderer
{
    public:
    SparkRenderer(const SparkRenderer&) = delete;
    SparkRenderer operator=(const SparkRenderer&) = delete;

    void setup();
    void renderPass();
    void cleanup();

    static SparkRenderer* getInstance();
    void updateBufferBindings() const;
    void drawGui();
    void addRenderingRequest(const RenderingRequest& request);

    private:
    std::map<ShaderType, std::deque<RenderingRequest>> renderQueue;

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
    std::shared_ptr<resources::Shader> medianFilterShader{nullptr};
    std::shared_ptr<resources::Shader> bloomDownScaleShader{nullptr};
    std::shared_ptr<resources::Shader> bloomUpScaleShader{nullptr};
    std::shared_ptr<resources::Shader> tileBasedLightCullingShader{nullptr};

    Cube cube = Cube();
    UniformBuffer cameraUBO{};
    UniformBuffer sampleUniformBuffer{};
    SSBO luminanceHistogram{};
    SSBO pointLightIndices{};
    SSBO spotLightIndices{};
    SSBO lightProbeIndices{};

    bool ssaoEnable = false;
    int kernelSize = 24;
    float radius = 0.45f;
    float bias = 0.015f;
    float power = 5.0f;

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
    bool renderingToCubemap = false;

    SparkRenderer() = default;
    ~SparkRenderer() = default;

    void resizeWindowIfNecessary();
    void fillGBuffer(const GBuffer& geometryBuffer);
    void fillGBuffer(const GBuffer& geometryBuffer, const std::function<bool(const RenderingRequest& request)>& filter);
    void ssaoComputing(const GBuffer& geometryBuffer);
    void renderLights(GLuint framebuffer, const GBuffer& geometryBuffer);
    void renderCubemap(GLuint framebuffer) const;
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
    void generateLightProbe(const std::shared_ptr<LightProbe>& lightProbe);
    void renderSceneToCubemap(const GBuffer& geometryBuffer, GLuint lightFbo, GLuint skyboxFbo);
};
}  // namespace spark
#endif