#ifndef SPARK_RENDERER_H
#define SPARK_RENDERER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Enums.h"
#include "ModelMesh.h"
#include "Structs.h"
#include "Shader.h"

namespace spark
{
class BlurPass;
class DepthOfFieldPass;

class SparkRenderer
{
    public:
    std::map<ShaderType, std::list<std::function<void(std::shared_ptr<resources::Shader>&)>>> renderQueue;

    SparkRenderer(const SparkRenderer&) = delete;
    SparkRenderer operator=(const SparkRenderer&) = delete;

    void setup();
    void renderPass();
    void cleanup();

    static SparkRenderer* getInstance();
    void updateBufferBindings() const;
    void drawGui();

    private:
    SparkRenderer() = default;
    ~SparkRenderer() = default;

    ScreenQuad screenQuad{};
    std::unique_ptr<DepthOfFieldPass> dofPass;
    std::unique_ptr<BlurPass> ssaoBlurPass;

    GLuint mainFramebuffer{}, colorTexture{}, normalsTexture{}, roughnessMetalnessTexture{}, depthTexture{};
    GLuint lightFrameBuffer{}, lightColorTexture{}, brightPassTexture{};
    GLuint cubemapFramebuffer{};
    GLuint toneMappingFramebuffer{}, toneMappingTexture{};
    GLuint motionBlurFramebuffer{}, motionBlurTexture{};
    GLuint lightShaftFramebuffer{}, lightShaftTexture{};
    GLuint fxaaFramebuffer{}, fxaaTexture{};
    GLuint averageLuminance{};

    //IBL
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
    std::shared_ptr<resources::Shader> downscaleShader{nullptr};

    Cube cube = Cube();
    UniformBuffer cameraUBO{};
    UniformBuffer sampleUniformBuffer{};
    SSBO luminanceHistogram{};

    bool ssaoEnable = true;
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
    float minLogLuminance = -0.5f;
    float oneOverLogLuminanceRange = 1.0f / 12.0f;
    float logLuminanceRange = 12.0f;
    float tau = 1.1f;

    bool bloomEnable = true;
    float bloomIntensity = 0.2f;

    bool motionBlurEnable = true;

    void resizeWindowIfNecessary();
    void fillGBuffer();
    void ssaoComputing();
    void renderLights();
    void renderCubemap() const;
    void bloom();
    void lightShafts();
    void helperShapes();
    void depthOfField();
    void fxaa();
    void motionBlur();
    void toneMapping();
    void calculateAverageLuminance();
    void renderToScreen() const;
    void initMembers();
    void createFrameBuffersAndTextures();
    void deleteFrameBuffersAndTextures() const;

    static void enableWireframeMode();
    static void disableWireframeMode();
};
}  // namespace spark
#endif