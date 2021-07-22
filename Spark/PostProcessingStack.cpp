#include "PostProcessingStack.hpp"

#include "CommonUtils.h"
#include "Spark.h"

namespace spark
{
PostProcessingStack::~PostProcessingStack()
{
    cleanup();
}

void PostProcessingStack::setup(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo)
{
    screenQuad.setup();
    ao.setup(width, height, cameraUbo);
    toneMapper.setup(width, height);
    bloomPass.setup(width, height);
    dofPass.setup(width, height, cameraUbo);
    lightShaftsPass.setup(width, height);
    skyboxPass.setup(width, height, cameraUbo);
    motionBlurPass.setup(width, height, cameraUbo);

    fxaaShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("fxaa.glsl");
    fxaaShader->use();
    fxaaShader->setVec2("inversedScreenSize", {1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height)});
}

GLuint PostProcessingStack::processAmbientOcclusion(GLuint depthTexture, GLuint normalsTexture)
{
    if(!isAmbientOcclusionEnabled)
        return 0;

    return ao.process(depthTexture, normalsTexture);
}

GLuint PostProcessingStack::process(GLuint lightingTexture, GLuint depthTexture, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap,
                                    const std::shared_ptr<Camera>& camera)
{
    textureHandle = lightingTexture;
    renderCubemap(lightingTexture, depthTexture, pbrCubemap);
    depthOfField(depthTexture);
    lightShafts(depthTexture, camera);
    motionBlur(depthTexture, camera);
    bloom(lightingTexture);
    toneMapping();
    fxaa();

    return textureHandle;
}

void PostProcessingStack::createFrameBuffersAndTextures(unsigned int width, unsigned int height)
{
    ao.createFrameBuffersAndTextures(width, height);
    dofPass.createFrameBuffersAndTextures(width, height);
    toneMapper.createFrameBuffersAndTextures(width, height);
    bloomPass.createFrameBuffersAndTextures(width, height);
    lightShaftsPass.createFrameBuffersAndTextures(width, height);
    skyboxPass.createFrameBuffersAndTextures(width, height);
    motionBlurPass.createFrameBuffersAndTextures(width, height);

    utils::recreateTexture2D(fxaaTexture, width, height, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(fxaaFramebuffer, {fxaaTexture});

    fxaaShader->use();
    fxaaShader->setVec2("inversedScreenSize", {1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height)});
}

void PostProcessingStack::cleanup()
{
    ao.cleanup();
    toneMapper.cleanup();
    bloomPass.cleanup();
    lightShaftsPass.cleanup();
    skyboxPass.cleanup();
    motionBlurPass.cleanup();

    glDeleteTextures(1, &fxaaTexture);
    glDeleteFramebuffers(1, &fxaaFramebuffer);
}

void PostProcessingStack::drawGui()
{
    if(ImGui::BeginMenu("Effects"))
    {
        const std::string menuName = "SSAO";
        if(ImGui::BeginMenu(menuName.c_str()))
        {
            ImGui::Checkbox("SSAO enabled", &isAmbientOcclusionEnabled);
            ImGui::DragInt("Samples", &ao.kernelSize, 1, 0, 64);
            ImGui::DragFloat("Radius", &ao.radius, 0.05f, 0.0f);
            ImGui::DragFloat("Bias", &ao.bias, 0.005f);
            ImGui::DragFloat("Power", &ao.power, 0.05f, 0.0f);
            ImGui::EndMenu();
        }

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
        ImGui::EndMenu();
    }
}

void PostProcessingStack::renderCubemap(GLuint lightingTexture, GLuint depthTexture, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap)
{
    if(auto outputOpt = skyboxPass.process(pbrCubemap, depthTexture, lightingTexture); outputOpt.has_value())
    {
        textureHandle = outputOpt.value();
    }
}

void PostProcessingStack::depthOfField(GLuint depthTexture)
{
    if(!isDofEnabled)
        return;

    dofPass.render(textureHandle, depthTexture);
    textureHandle = dofPass.getOutputTexture();
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
    PUSH_DEBUG_GROUP(FXAA);

    glBindFramebuffer(GL_FRAMEBUFFER, fxaaFramebuffer);

    fxaaShader->use();

    glBindTextureUnit(0, textureHandle);
    screenQuad.draw();
    glBindTextures(0, 2, nullptr);

    textureHandle = fxaaTexture;
    POP_DEBUG_GROUP();
}
}  // namespace spark