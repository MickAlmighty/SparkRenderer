#include "PostProcessingStack.hpp"

#include "CommonUtils.h"
#include "Spark.h"

namespace spark::effects
{
PostProcessingStack::PostProcessingStack(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo)
    : toneMapper(width, height), bloomPass(width, height), dofPass(width, height, cameraUbo), lightShaftsPass(width, height),
      skyboxPass(width, height), motionBlurPass(width, height, cameraUbo), fxaaPass(width, height)
{
}

GLuint PostProcessingStack::process(GLuint lightingTexture, GLuint depthTexture, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap,
                                    const std::shared_ptr<Camera>& camera, const UniformBuffer& cameraUbo)
{
    textureHandle = lightingTexture;
    renderCubemap(lightingTexture, depthTexture, pbrCubemap, cameraUbo);
    depthOfField(depthTexture);
    lightShafts(depthTexture, camera);
    motionBlur(depthTexture, camera);
    bloom(lightingTexture);
    toneMapping();
    fxaa();

    return textureHandle;
}

void PostProcessingStack::resize(unsigned int width, unsigned int height)
{
    dofPass.resize(width, height);
    toneMapper.resize(width, height);
    bloomPass.resize(width, height);
    lightShaftsPass.resize(width, height);
    skyboxPass.resize(width, height);
    motionBlurPass.resize(width, height);
    fxaaPass.resize(width, height);
}

void PostProcessingStack::drawGui()
{
    const std::string menuName2 = "Depth of Field";
    if(ImGui::BeginMenu(menuName2.c_str()))
    {
        ImGui::Checkbox("DOF enabled", &isDofEnabled);
        ImGui::DragFloat("NearStart", &dofPass.nearStart, 0.1f, 0.0f);
        ImGui::DragFloat("NearEnd", &dofPass.nearEnd, 0.1f);
        ImGui::DragFloat("FarStart", &dofPass.farStart, 0.1f, 0.0f);
        ImGui::DragFloat("FarEnd", &dofPass.farEnd, 0.1f, 0.0f);
        ImGui::EndMenu();
    }

    const std::string menuName3 = "Light Shafts";
    if(ImGui::BeginMenu(menuName3.c_str()))
    {
        ImGui::Checkbox("Light shafts enabled", &isLightShaftsPassEnabled);
        ImGui::DragFloat("Exposure", &lightShaftsPass.exposure, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat("Decay", &lightShaftsPass.decay, 0.01f, 0.0001f);
        ImGui::DragFloat("Density", &lightShaftsPass.density, 0.01f, 0.0001f, 1.0f);
        ImGui::DragFloat("Weight", &lightShaftsPass.weight, 0.01f, 0.0001f);
        ImGui::EndMenu();
    }

    const std::string menuName4 = "Tone Mapping";
    if(ImGui::BeginMenu(menuName4.c_str()))
    {
        ImGui::DragFloat("minLogLuminance", &toneMapper.minLogLuminance, 0.01f);
        ImGui::DragFloat("logLuminanceRange", &toneMapper.logLuminanceRange, 0.01f);
        ImGui::DragFloat("tau", &toneMapper.tau, 0.01f, 0.0f);
        ImGui::EndMenu();
    }

    const std::string menuName5 = "Bloom";
    if(ImGui::BeginMenu(menuName5.c_str()))
    {
        ImGui::Checkbox("Bloom", &isBloomEnabled);
        ImGui::DragFloat("Intensity", &bloomPass.intensity, 0.001f, 0);
        ImGui::DragFloat("Threshold", &bloomPass.threshold, 0.01f, 0.0001f);
        ImGui::DragFloat("ThresholdSize", &bloomPass.thresholdSize, 0.01f, 0.0001f);
        ImGui::DragFloat("radiusMip0", &bloomPass.radiusMip0, 0.01f, 0.0001f);
        ImGui::DragFloat("radiusMip1", &bloomPass.radiusMip1, 0.01f, 0.0001f);
        ImGui::DragFloat("radiusMip2", &bloomPass.radiusMip2, 0.01f, 0.0001f);
        ImGui::DragFloat("radiusMip3", &bloomPass.radiusMip3, 0.01f, 0.0001f);
        ImGui::DragFloat("radiusMip4", &bloomPass.radiusMip4, 0.01f, 0.0001f);
        ImGui::EndMenu();
    }

    const std::string menuName6 = "MotionBlur";
    if(ImGui::BeginMenu(menuName6.c_str()))
    {
        ImGui::Checkbox("Motion Blur", &isMotionBlurEnabled);
        ImGui::EndMenu();
    }
}

void PostProcessingStack::renderCubemap(GLuint lightingTexture, GLuint depthTexture, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap,
                                        const UniformBuffer& cameraUbo)
{
    if(auto outputOpt = skyboxPass.process(pbrCubemap, depthTexture, lightingTexture, cameraUbo); outputOpt.has_value())
    {
        textureHandle = outputOpt.value();
    }
}

void PostProcessingStack::depthOfField(GLuint depthTexture)
{
    if(!isDofEnabled)
        return;

    textureHandle = dofPass.process(textureHandle, depthTexture);
}

void PostProcessingStack::lightShafts(GLuint depthTexture, const std::shared_ptr<Camera>& camera)
{
    if(isLightShaftsPassEnabled)
    {
        if(auto outputOpt = lightShaftsPass.process(camera, depthTexture, textureHandle); outputOpt.has_value())
        {
            textureHandle = outputOpt.value();
        }
    }
}

void PostProcessingStack::bloom(GLuint lightingTexture)
{
    if(isBloomEnabled)
    {
        textureHandle = bloomPass.process(textureHandle, lightingTexture);
    }
}

void PostProcessingStack::motionBlur(GLuint depthTexture, const std::shared_ptr<Camera>& camera)
{
    if(auto outputTextureOpt = motionBlurPass.process(camera, textureHandle, depthTexture); outputTextureOpt.has_value() && isMotionBlurEnabled)
    {
        textureHandle = outputTextureOpt.value();
    }
}

void PostProcessingStack::toneMapping()
{
    textureHandle = toneMapper.process(textureHandle);
}

void PostProcessingStack::fxaa()
{
    textureHandle = fxaaPass.process(textureHandle);
}
}  // namespace spark::effects