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
    GLuint lightFrameBuffer{}, lightColorTexture{};
    GLuint toneMappingFramebuffer{}, toneMappingTexture{};
    GLuint motionBlurFramebuffer{}, motionBlurTexture{};
    GLuint lightShaftFramebuffer{}, lightShaftTexture{};
    GLuint fxaaFramebuffer{}, fxaaTexture{};

    GLuint ssaoFramebuffer{}, ssaoTexture{}, randomNormalsTexture{};

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
    Cube cube = Cube();
    UniformBuffer cameraUBO{};
    UniformBuffer sampleUniformBuffer{};

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

    void resizeWindowIfNecessary();
    void fillGBuffer();
    void ssaoComputing() const;
    void renderLights();
    void renderCubemap() const;
    void depthOfField();
    void lightShafts();
    void fxaa();
    void motionBlur();
    void toneMapping();
    void renderToScreen() const;
    void initMembers();
    void createFrameBuffersAndTextures();
    void deleteFrameBuffersAndTextures() const;
};
}  // namespace spark
#endif