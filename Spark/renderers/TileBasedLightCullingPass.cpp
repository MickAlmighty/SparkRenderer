#include "TileBasedLightCullingPass.hpp"

#include "EditorCamera.hpp"
#include "utils/CommonUtils.h"
#include "Scene.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::renderers
{
TileBasedLightCullingPass::TileBasedLightCullingPass(unsigned int width, unsigned int height) : w(width), h(height)
{
    tileBasedLightCullingShader =
        Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/tileBasedLightCulling.glsl");

    createFrameBuffersAndTextures();
}

void TileBasedLightCullingPass::process(GLuint depthTexture, const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    PUSH_DEBUG_GROUP(TILE_BASED_LIGHTS_CULLING);

    tileBasedLightCullingShader->use();
    tileBasedLightCullingShader->bindUniformBuffer("Camera", camera->getUbo());

    tileBasedLightCullingShader->bindSSBO("PointLightIndices", pointLightIndices);
    tileBasedLightCullingShader->bindSSBO("SpotLightIndices", spotLightIndices);
    tileBasedLightCullingShader->bindSSBO("LightProbeIndices", lightProbeIndices);

    bindLightBuffers(scene->lightManager);

    glBindTextureUnit(0, depthTexture);

    tileBasedLightCullingShader->dispatchCompute(utils::uiCeil(w, 16u), utils::uiCeil(h, 16u), 1);

    POP_DEBUG_GROUP();
}

void TileBasedLightCullingPass::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    createFrameBuffersAndTextures();
}

void TileBasedLightCullingPass::createFrameBuffersAndTextures()
{
    constexpr unsigned int lightCount = 4096;
    pointLightIndices.resizeBuffer(lightCount * utils::uiCeil(h, 16u) * utils::uiCeil(w, 16u) * sizeof(uint32_t));
    spotLightIndices.resizeBuffer(lightCount * utils::uiCeil(h, 16u) * utils::uiCeil(w, 16u) * sizeof(uint32_t));
    lightProbeIndices.resizeBuffer(lightCount * utils::uiCeil(h, 16u) * utils::uiCeil(w, 16u) * sizeof(uint32_t));
}

void TileBasedLightCullingPass::bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager)
{
    tileBasedLightCullingShader->bindSSBO("DirLightData", lightManager->getDirLightSSBO());
    tileBasedLightCullingShader->bindSSBO("PointLightData", lightManager->getPointLightSSBO());
    tileBasedLightCullingShader->bindSSBO("SpotLightData", lightManager->getSpotLightSSBO());
    tileBasedLightCullingShader->bindSSBO("LightProbeData", lightManager->getLightProbeSSBO());
}
}  // namespace spark::renderers