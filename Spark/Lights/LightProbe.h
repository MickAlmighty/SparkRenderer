#pragma once

#include "Component.h"

namespace spark
{
struct LightProbeData;
class Mesh;

class LightProbe : public Component
{
    public:
    bool generateLightProbe{true};

    LightProbe();
    ~LightProbe();

    LightProbe(const LightProbe&) = delete;
    LightProbe(const LightProbe&&) = delete;
    LightProbe& operator=(const LightProbe&) = delete;
    LightProbe& operator=(const LightProbe&&) = delete;
    bool operator<(const LightProbe& lightProbe) const;

    void update() override;
    void fixedUpdate() override;
    void drawGUI() override;

    [[nodiscard]] LightProbeData getLightData() const;
    [[nodiscard]] bool getDirty() const;
    [[nodiscard]] float getRadius() const;
    [[nodiscard]] float getFadeDistance() const;
    void resetDirty();
    [[nodiscard]] GLuint getPrefilterCubemap() const;
    [[nodiscard]] GLuint getIrradianceCubemap() const;

    void renderIntoIrradianceCubemap(GLuint framebuffer, GLuint environmentCubemap, Cube& cube,
                                     const std::shared_ptr<resources::Shader>& irradianceShader) const;
    void renderIntoPrefilterCubemap(GLuint framebuffer, GLuint environmentCubemap, unsigned envCubemapSize, Cube& cube,
                                    const std::shared_ptr<resources::Shader>& prefilterShader,
                                    const std::shared_ptr<resources::Shader>& resampleCubemapShader) const;

    void setActive(bool active_) override;
    void setRadius(float radius_);
    void setFadeDistance(float fadeDistance_);
    // void setIrradianceCubemap(GLuint irradianceCubemap_);
    // void setPrefilterCubemap(GLuint prefilterCubemap_);

    private:
    GLuint irradianceCubemap{};
    GLuint prefilterCubemap{};
    GLuint64 irradianceCubemapHandle{};
    GLuint64 prefilterCubemapHandle{};
    float radius{1};
    float fadeDistance{1};

    GLuint irradianceCubemapSize = 32;
    GLuint prefilterCubemapSize = 128;

    bool dirty{true};
    bool addedToLightManager{false};

    std::shared_ptr<Mesh> sphere{nullptr};

    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(Component);
};

}  // namespace spark
