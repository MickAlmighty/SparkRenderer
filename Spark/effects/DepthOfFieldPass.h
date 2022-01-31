#pragma once

#include <memory>

#include "Buffer.hpp"
#include "BlurPass.h"
#include "glad_glfw3.h"
#include "ScreenQuad.hpp"
#include "utils/GlHandle.hpp"

namespace spark
{
class ICamera;
}

namespace spark::resources
{
class Shader;
}

namespace spark::effects
{
class BlurPass;

class DepthOfFieldPass
{
    public:
    GLuint process(GLuint lightPassTexture, GLuint depthTexture, const std::shared_ptr<ICamera>& camera) const;
    void resize(unsigned int width, unsigned int height);

    DepthOfFieldPass(unsigned int width, unsigned int height);
    DepthOfFieldPass& operator=(const DepthOfFieldPass& blurPass) = delete;
    DepthOfFieldPass& operator=(const DepthOfFieldPass&& blurPass) = delete;
    DepthOfFieldPass(const DepthOfFieldPass& blurPass) = delete;
    DepthOfFieldPass(const DepthOfFieldPass&& blurPass) = delete;
    ~DepthOfFieldPass();

    float nearStart{1}, nearEnd{4};
    float farStart{20}, farEnd{100};

    private:
    void createFrameBuffersAndTextures();

    void calculateCircleOfConfusion(GLuint depthTexture, const std::shared_ptr<ICamera>& camera) const;
    void blurLightPassTexture(GLuint lightPassTexture) const;
    inline void detectBokehPositions(GLuint lightPassTexture) const;
    inline void renderBokehShapes() const;
    void blendDepthOfField(GLuint lightPassTexture) const;

    unsigned int w{}, h{};

    std::shared_ptr<resources::Shader> cocShader{nullptr};
    std::shared_ptr<resources::Shader> blendShader{nullptr};

    /*GLuint indirectBufferID{}, bokehCountTexID{}, bokehCounterID{};
    SSBO bokehPositionBuffer, bokehColorBuffer;
    GLuint bokehPositionTexture{}, bokehColorTexture{};*/

    utils::TextureHandle cocTexture{}, blendDofTexture{};
    GLuint cocFramebuffer{}, blendDofFramebuffer{};

    BlurPass blurPass;
    ScreenQuad screenQuad{};
};
}  // namespace spark::effects