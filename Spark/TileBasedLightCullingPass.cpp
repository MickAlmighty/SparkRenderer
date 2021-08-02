#include "TileBasedLightCullingPass.hpp"

#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark
{
TileBasedLightCullingPass::TileBasedLightCullingPass(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
                                                     const std::shared_ptr<lights::LightManager>& lightManager)
    : w(width), h(height)
{
    tileBasedLightCullingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("tileBasedLightCulling.glsl");
    tileBasedLightCullingShader->bindUniformBuffer("Camera", cameraUbo);

    tileBasedLightCullingShader->bindSSBO("PointLightIndices", pointLightIndices);
    tileBasedLightCullingShader->bindSSBO("SpotLightIndices", spotLightIndices);
    tileBasedLightCullingShader->bindSSBO("LightProbeIndices", lightProbeIndices);
    bindLightBuffers(lightManager);
    createFrameBuffersAndTextures();
}

TileBasedLightCullingPass::~TileBasedLightCullingPass()
{
    glDeleteTextures(1, &lightsPerTileTexture);
}

void TileBasedLightCullingPass::process(GLuint depthTexture)
{
    PUSH_DEBUG_GROUP(TILE_BASED_LIGHTS_CULLING);
    pointLightIndices.clearData();
    spotLightIndices.clearData();
    lightProbeIndices.clearData();

    tileBasedLightCullingShader->use();

    glBindTextureUnit(0, depthTexture);

    // debug of light count per tile
    /*uint8_t clear[]{0,0,0,0};
    glClearTexImage(lightsPerTileTexture, 0, GL_RGBA, GL_UNSIGNED_BYTE, &clear);*/
    glBindImageTexture(5, lightsPerTileTexture, 0, false, 0, GL_READ_WRITE, GL_RGBA16F);

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
    pointLightIndices.resizeBuffer(256 * utils::uiCeil(h, 16u) * utils::uiCeil(w, 16u) * sizeof(uint32_t));
    spotLightIndices.resizeBuffer(256 * utils::uiCeil(h, 16u) * utils::uiCeil(w, 16u) * sizeof(uint32_t));
    lightProbeIndices.resizeBuffer(256 * utils::uiCeil(h, 16u) * utils::uiCeil(w, 16u) * sizeof(uint32_t));
    utils::recreateTexture2D(lightsPerTileTexture, w / 16, h / 16, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
}

void TileBasedLightCullingPass::bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager)
{
    tileBasedLightCullingShader->bindSSBO("DirLightData", lightManager->getDirLightSSBO());
    tileBasedLightCullingShader->bindSSBO("PointLightData", lightManager->getPointLightSSBO());
    tileBasedLightCullingShader->bindSSBO("SpotLightData", lightManager->getSpotLightSSBO());
    tileBasedLightCullingShader->bindSSBO("LightProbeData", lightManager->getLightProbeSSBO());
}
}  // namespace spark