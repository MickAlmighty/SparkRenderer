#pragma once

#include <glad/glad.h>

#include "AmbientOcclusion.hpp"
#include "Enums.h"
#include "GBuffer.h"
#include "PostProcessingStack.hpp"
#include "Scene.h"
#include "ScreenQuad.hpp"
#include "RenderingRequest.h"

namespace spark
{
class LightProbe;
struct PbrCubemapTexture;

class SparkRenderer
{
    public:
    SparkRenderer(const SparkRenderer&) = delete;
    SparkRenderer(SparkRenderer&&) = delete;
    SparkRenderer& operator=(const SparkRenderer&) = delete;
    SparkRenderer& operator=(SparkRenderer&&) = delete;
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
    void renderLights(GLuint framebuffer, const GBuffer& geometryBuffer);
    void tileBasedLightCulling(const GBuffer& geometryBuffer) const;
    void tileBasedLightRendering(const GBuffer& geometryBuffer, GLuint ssaoTexture);

    void helperShapes();
    void renderToScreen();
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

    GBuffer gBuffer{};

    GLuint uiShapesFramebuffer{};
    GLuint lightFrameBuffer{}, lightingTexture{};
    GLuint lightsPerTileTexture{};

    // IBL
    GLuint brdfLookupTexture{};

    // local light probes start
    const unsigned int sceneCubemapSize{256};
    GLuint lightProbeSceneCubemap{};
    GLuint lightProbeLightFbo{};
    GLuint lightProbeSkyboxFbo{};
    GBuffer localLightProbeGBuffer{};
    SSBO cubemapViewMatrices{};
    // local light probes end

    GLuint textureHandle{};  // temporary, its only a handle to other texture -> dont delete it

    std::shared_ptr<resources::Shader> mainShader{nullptr};
    std::shared_ptr<resources::Shader> screenShader{nullptr};
    std::shared_ptr<resources::Shader> lightShader{nullptr};
    std::shared_ptr<resources::Shader> solidColorShader{nullptr};
    std::shared_ptr<resources::Shader> tileBasedLightCullingShader{nullptr};
    std::shared_ptr<resources::Shader> tileBasedLightingShader{nullptr};
    std::shared_ptr<resources::Shader> localLightProbesLightingShader{nullptr};
    std::shared_ptr<resources::Shader> equirectangularToCubemapShader{nullptr};
    std::shared_ptr<resources::Shader> irradianceShader{nullptr};
    std::shared_ptr<resources::Shader> prefilterShader{nullptr};
    std::shared_ptr<resources::Shader> resampleCubemapShader{nullptr};

    PostProcessingStack postProcessingStack{};
    SkyboxPass skyboxPass{};

    Cube cube = Cube();
    UniformBuffer cameraUBO{};
    SSBO pointLightIndices{};
    SSBO spotLightIndices{};
    SSBO lightProbeIndices{};
};
}  // namespace spark