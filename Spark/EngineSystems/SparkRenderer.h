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
    std::map<ShaderType, std::list<std::function<void(std::shared_ptr<Shader>&)>>> renderQueue;

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

    std::weak_ptr<Shader> mainShader;
    std::weak_ptr<Shader> screenShader;
    std::weak_ptr<Shader> toneMappingShader;
    std::weak_ptr<Shader> lightShader;
    std::weak_ptr<Shader> motionBlurShader;
    std::weak_ptr<Shader> cubemapShader;
    std::weak_ptr<Shader> ssaoShader;
    std::weak_ptr<Shader> circleOfConfusionShader;
    std::weak_ptr<Shader> bokehDetectionShader;
    std::weak_ptr<Shader> blendDofShader;
    std::weak_ptr<Shader> solidColorShader;
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