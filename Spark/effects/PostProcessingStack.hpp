#pragma once

#include "AmbientOcclusion.hpp"
#include "BloomPass.hpp"
#include "Buffer.hpp"
#include "DepthOfFieldPass.h"
#include "FxaaPass.hpp"
#include "glad_glfw3.h"
#include "LightShaftsPass.hpp"
#include "MotionBlurPass.hpp"
#include "SkyboxPass.hpp"
#include "ToneMapper.hpp"

namespace spark::effects
{
class PostProcessingStack
{
    public:
    PostProcessingStack() = default;
    PostProcessingStack(const PostProcessingStack&) = delete;
    PostProcessingStack(PostProcessingStack&&) = delete;
    PostProcessingStack& operator=(const PostProcessingStack&) = delete;
    PostProcessingStack& operator=(PostProcessingStack&&) = delete;
    ~PostProcessingStack();

    void setup(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo);
    GLuint process(GLuint lightingTexture, GLuint depthTexture, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap,
                   const std::shared_ptr<Camera>& camera, const UniformBuffer& cameraUbo);
    void createFrameBuffersAndTextures(unsigned int width, unsigned int height);
    void cleanup();

    void drawGui();

    private:
    void renderCubemap(GLuint lightingTexture, GLuint depthTexture, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap,
                       const UniformBuffer& cameraUbo);
    void depthOfField(GLuint depthTexture);
    void lightShafts(GLuint depthTexture, const std::shared_ptr<Camera>& camera);
    void bloom(GLuint lightingTexture);
    void motionBlur(GLuint depthTexture, const std::shared_ptr<Camera>& camera);
    void toneMapping();
    void fxaa();

    bool isDofEnabled = false;
    bool isBloomEnabled = false;
    bool isLightShaftsPassEnabled = true;
    bool isMotionBlurEnabled = true;

    GLuint textureHandle{};

    ToneMapper toneMapper{};
    BloomPass bloomPass{};
    DepthOfFieldPass dofPass{};
    LightShaftsPass lightShaftsPass{};
    SkyboxPass skyboxPass{};
    MotionBlurPass motionBlurPass{};
    FxaaPass fxaaPass{};
};
}  // namespace spark::effects