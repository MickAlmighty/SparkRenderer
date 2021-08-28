#pragma once

#include <glm/glm.hpp>

#include "Component.h"
#include "Cube.hpp"
#include "glad_glfw3.h"
#include "LightManager.h"
#include "LightStatus.hpp"
#include "Observable.hpp"
#include "Shader.h"

namespace spark
{
class Mesh;
}

namespace spark::lights
{
struct LightProbeData final
{
    GLint64 irradianceCubemapHandle{0};
    GLint64 prefilterCubemapHandle{0};
    glm::vec4 positionAndRadius{0};
    alignas(16) float fadeDistance{0};

    bool operator<(const LightProbeData& rhs) const
    {
        return positionAndRadius.w < rhs.positionAndRadius.w;
    }
};

class LightProbe : public Component, public Observable<LightStatus<LightProbe>>
{
    public:
    bool generateLightProbe{true};

    LightProbe();
    ~LightProbe() override;

    LightProbe(const LightProbe&) = delete;
    LightProbe(const LightProbe&&) = delete;
    LightProbe& operator=(const LightProbe&) = delete;
    LightProbe& operator=(const LightProbe&&) = delete;

    void update() override;
    void fixedUpdate() override;
    void drawGUI() override;

    [[nodiscard]] LightProbeData getLightData() const;
    [[nodiscard]] float getRadius() const;
    [[nodiscard]] float getFadeDistance() const;
    [[nodiscard]] GLuint getPrefilterCubemap() const;
    [[nodiscard]] GLuint getIrradianceCubemap() const;

    void renderIntoIrradianceCubemap(GLuint framebuffer, GLuint environmentCubemap, Cube& cube,
                                     const std::shared_ptr<resources::Shader>& irradianceShader) const;
    void renderIntoPrefilterCubemap(GLuint framebuffer, GLuint environmentCubemap, unsigned envCubemapSize, Cube& cube,
                                    const std::shared_ptr<resources::Shader>& prefilterShader,
                                    const std::shared_ptr<resources::Shader>& resampleCubemapShader) const;
    void setRadius(float radius_);
    void setFadeDistance(float fadeDistance_);

    private:
    void onActive() override;
    void onInactive() override;
    void notifyAbout(LightCommand command);

    GLuint irradianceCubemap{};
    GLuint prefilterCubemap{};
    GLuint64 irradianceCubemapHandle{};
    GLuint64 prefilterCubemapHandle{};
    glm::vec3 position{0.0f};
    float radius{1};
    float fadeDistance{1};
    std::shared_ptr<LightManager> lightManager{nullptr};

    const GLuint irradianceCubemapSize = 32;
    const GLuint prefilterCubemapSize = 128;

    std::shared_ptr<Mesh> sphere{nullptr};

    RTTR_ENABLE(Component)
};
}  // namespace spark::lights
