#pragma once

#include "BloomPass.hpp"
#include "DepthOfFieldPass.h"
#include "FxaaPass.hpp"
#include "glad_glfw3.h"
#include "LightShaftsPass.hpp"
#include "MotionBlurPass.hpp"
#include "Scene.h"
#include "SkyboxPass.hpp"
#include "ToneMapper.hpp"

namespace spark::effects
{
class PostProcessingStack
{
    public:
    PostProcessingStack(unsigned int width, unsigned int height);
    PostProcessingStack(const PostProcessingStack&) = delete;
    PostProcessingStack(PostProcessingStack&&) = delete;
    PostProcessingStack& operator=(const PostProcessingStack&) = delete;
    PostProcessingStack& operator=(PostProcessingStack&&) = delete;
    ~PostProcessingStack() = default;

    GLuint process(GLuint lightingTexture, GLuint depthTexture, const std::shared_ptr<Scene>& scene);
    void resize(unsigned int width, unsigned int height);

    void drawGui();

    private:
    void renderCubemap(GLuint lightingTexture, GLuint depthTexture, const std::shared_ptr<Scene>& scene);
    void depthOfField(GLuint depthTexture, const std::shared_ptr<Camera>& camera);
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

    ToneMapper toneMapper;
    BloomPass bloomPass;
    DepthOfFieldPass dofPass;
    LightShaftsPass lightShaftsPass;
    SkyboxPass skyboxPass;
    MotionBlurPass motionBlurPass;
    FxaaPass fxaaPass;
};
}  // namespace spark::effects