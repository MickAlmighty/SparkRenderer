#pragma once

#include <set>

#include "utils/CommonUtils.h"

namespace spark::lights
{
struct LightProbeCubemaps
{
    utils::UniqueTextureHandle irradianceCubemap;
    utils::UniqueTextureHandle prefilterCubemap;
    unsigned int id;
};

class LightProbeManager
{
    public:
    LightProbeManager(unsigned int numberOfLightProbes);

    LightProbeCubemaps acquireLightProbeCubemaps();
    void releaseLightProbeCubemaps(const LightProbeCubemaps& lightProbeData);

    GLuint getPrefilterCubemapArray() const;
    GLuint getIrradianceCubemapArray() const;

    private:
    unsigned int getFreeId();

    const GLuint irradianceCubemapSize = 32;
    const GLuint prefilterCubemapSize = 128;
    const GLuint lightProbesCount = 256;

    utils::UniqueTextureHandle prefilterCubemapArray{};
    utils::UniqueTextureHandle irradianceCubemapArray{};

    std::set<unsigned int> acquiredIds;
};
}  // namespace spark::lights