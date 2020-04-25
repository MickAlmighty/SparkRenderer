#pragma once
#include "Component.h"

namespace spark
{
class LightProbe : public Component
{
    public:
    bool generateLightProbe{true};

    LightProbe();
    ~LightProbe() = default;

    LightProbe(const LightProbe&) = delete;
    LightProbe(const LightProbe&&) = delete;
    LightProbe& operator=(const LightProbe&) = delete;
    LightProbe& operator=(const LightProbe&&) = delete;

    void update() override;
    void fixedUpdate() override;
    void drawGUI() override;

    GLuint getPrefilterCubemap() const;
    GLuint getIrradianceCubemap() const;

    void setIrradianceCubemap(GLuint irradianceCubemap_);
    void setPrefilterCubemap(GLuint prefilterCubemap_);

    private:
    GLuint irradianceCubemap{};
    GLuint prefilterCubemap{};

    bool addedToLightManager{false};

    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(Component);
};

}  // namespace spark
