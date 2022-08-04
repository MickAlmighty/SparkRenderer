#include "LightProbeManager.hpp"

#include <algorithm>

namespace spark::lights
{
LightProbeManager::LightProbeManager(unsigned int numberOfLightProbes)
{
    prefilterCubemapArray =
        utils::createCubemapArray(prefilterCubemapSize, prefilterCubemapSize, lightProbesCount, GL_R11F_G11F_B10F, GL_CLAMP_TO_EDGE, GL_LINEAR, 5);
    irradianceCubemapArray =
        utils::createCubemapArray(irradianceCubemapSize, irradianceCubemapSize, lightProbesCount, GL_R11F_G11F_B10F, GL_CLAMP_TO_EDGE, GL_LINEAR, 1);
}

GLuint LightProbeManager::getPrefilterCubemapArray() const
{
    return prefilterCubemapArray.get();
}

GLuint LightProbeManager::getIrradianceCubemapArray() const
{
    return irradianceCubemapArray.get();
}

LightProbeCubemaps LightProbeManager::acquireLightProbeCubemaps()
{
    auto id = getFreeId();

    GLuint irradianceCubemapView{0};
    glGenTextures(1, &irradianceCubemapView);
    glTextureView(irradianceCubemapView, GL_TEXTURE_CUBE_MAP, irradianceCubemapArray.get(), GL_R11F_G11F_B10F, 0, 1, 6 * id, 6);

    GLuint prefilterCubemapView{0};
    glGenTextures(1, &prefilterCubemapView);
    glTextureView(prefilterCubemapView, GL_TEXTURE_CUBE_MAP, prefilterCubemapArray.get(), GL_R11F_G11F_B10F, 0, 5, 6 * id, 6);

    return {irradianceCubemapView, prefilterCubemapView, id};
}

void LightProbeManager::releaseLightProbeCubemaps(const LightProbeCubemaps& lightProbeCubemaps)
{
    acquiredIds.erase(lightProbeCubemaps.id);
}

unsigned int LightProbeManager::getFreeId()
{
    if(acquiredIds.empty())
    {
        acquiredIds.insert(0);
        return 0;
    }

    const auto it = std::adjacent_find(acquiredIds.begin(), acquiredIds.end(), [](unsigned int lhs, unsigned int rhs) { return rhs - lhs > 1; });
    if(it != std::end(acquiredIds))
    {
        unsigned int id = *it + 1;
        acquiredIds.insert(id);
        return id;
    }

    unsigned int id = *std::prev(acquiredIds.end()) + 1;
    acquiredIds.insert(id);
    return id;
}
}  // namespace spark::lights