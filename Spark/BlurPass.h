#ifndef BLURR_PASS_H
#define BLURR_PASS_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include "Shader.h"
#include "Structs.h"

namespace spark
{

class BlurPass
{
    public:
    void blurTexture(GLuint texture) const;
    GLuint getBlurredTexture() const;
    GLuint getSecondPassFramebuffer() const;
    void recreateWithNewSize(unsigned int width, unsigned int height);

    BlurPass(unsigned int width_, unsigned int height_);
    ~BlurPass();

    BlurPass& operator=(const BlurPass& blurPass) = delete;
    BlurPass& operator=(const BlurPass&& blurPass) = delete;
    BlurPass(const BlurPass& blurPass) = delete;
    BlurPass(const BlurPass&& blurPass) = delete;

    private:
    unsigned int width{}, height{};
    GLuint hFramebuffer{}, hTexture{};
    GLuint vFramebuffer{}, vTexture{};
    ScreenQuad screenQuad{};

    std::shared_ptr<resources::Shader> gaussianBlurShader{ nullptr };

    void createGlObjects();
    void deleteGlObjects();
};
}  // namespace spark

#endif