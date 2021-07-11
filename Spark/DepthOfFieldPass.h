#pragma once

#include <memory>

#include "Buffer.hpp"
#include "glad_glfw3.h"
#include "ScreenQuad.hpp"

namespace spark
{
namespace resources
{
    class Shader;
}

class BlurPass;

class DepthOfFieldPass
{
    public:
    void setup(unsigned int width_, unsigned int height_, const UniformBuffer& cameraUbo);
    void render(GLuint lightPassTexture, GLuint depthTexture) const;
    GLuint getOutputTexture() const;
    void createFrameBuffersAndTextures(unsigned int width, unsigned int height);
    void setUniforms(float nearStart, float nearEnd, float farStart, float farEnd);

    DepthOfFieldPass() = default;
    DepthOfFieldPass& operator=(const DepthOfFieldPass& blurPass) = delete;
    DepthOfFieldPass& operator=(const DepthOfFieldPass&& blurPass) = delete;
    DepthOfFieldPass(const DepthOfFieldPass& blurPass) = delete;
    DepthOfFieldPass(const DepthOfFieldPass&& blurPass) = delete;
    ~DepthOfFieldPass();

    private:
    void calculateCircleOfConfusion(GLuint depthTexture) const;
    void blurLightPassTexture(GLuint lightPassTexture) const;
    inline void detectBokehPositions(GLuint lightPassTexture) const;
    inline void renderBokehShapes() const;
    void blendDepthOfField(GLuint lightPassTexture) const;

    void createGlObjects();
    void deleteGlObjects();

    unsigned int width{}, height{};

    float nearStart{ 1 }, nearEnd{ 4 };
    float farStart{ 20 }, farEnd{ 100 };

    std::shared_ptr<resources::Shader> cocShader{ nullptr };
    std::shared_ptr<resources::Shader> blendShader{ nullptr };

    /*GLuint indirectBufferID{}, bokehCountTexID{}, bokehCounterID{};
    SSBO bokehPositionBuffer, bokehColorBuffer;
    GLuint bokehPositionTexture{}, bokehColorTexture{};*/

    GLuint cocFramebuffer{}, cocTexture{};
    GLuint blendDofFramebuffer{}, blendDofTexture{};

    std::unique_ptr<BlurPass> blurPass;
    ScreenQuad screenQuad{};
};
}  // namespace spark