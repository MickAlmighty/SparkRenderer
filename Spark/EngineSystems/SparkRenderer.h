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

    GLuint mainFramebuffer{}, colorTexture{}, normalsTexture{}, depthTexture{};
    GLuint lightFrameBuffer{}, lightColorTexture{}, lightDiffuseTexture{}, lightSpecularTexture{};
    GLuint toneMappingFramebuffer{}, toneMappingTexture{};
    GLuint motionBlurFramebuffer{}, motionBlurTexture{};
    GLuint lightShaftFramebuffer{}, lightShaftTexture{};
    GLuint fxaaFramebuffer{}, fxaaTexture{};
    GLuint averageLuminance{};

    GLuint ssaoFramebuffer{}, ssaoTexture{}, randomNormalsTexture{}, ssaoDisabledTexture{};

    GLuint textureHandle{};  // temporary, its only a handle to other texture -> dont delete it

    std::shared_ptr<resources::Shader> mainShader{ nullptr };
    std::shared_ptr<resources::Shader> screenShader{ nullptr };
    std::shared_ptr<resources::Shader> toneMappingShader{ nullptr };
    std::shared_ptr<resources::Shader> lightShader{ nullptr };
    std::shared_ptr<resources::Shader> motionBlurShader{ nullptr };
    std::shared_ptr<resources::Shader> cubemapShader{ nullptr };
    std::shared_ptr<resources::Shader> ssaoShader{ nullptr };
    std::shared_ptr<resources::Shader> circleOfConfusionShader{ nullptr };
    std::shared_ptr<resources::Shader> bokehDetectionShader{ nullptr };
    std::shared_ptr<resources::Shader> blendDofShader{ nullptr };
    std::shared_ptr<resources::Shader> solidColorShader{ nullptr };
    std::shared_ptr<resources::Shader> lightShaftsShader{ nullptr };
    std::shared_ptr<resources::Shader> luminanceHistogramComputeShader{ nullptr };
    std::shared_ptr<resources::Shader> averageLuminanceComputeShader{ nullptr };
    std::shared_ptr<resources::Shader> fxaaShader{ nullptr };

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

    void resizeWindowIfNecessary();
    void fillGBuffer();
    void ssaoComputing();
    void renderLights();
    void renderCubemap() const;
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