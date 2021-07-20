#pragma once
#include <memory>

#include "glad_glfw3.h"
#include "ScreenQuad.hpp"

namespace spark
{
namespace resources {
    class Shader;
}

class TexturePass
{
    public:
    TexturePass();
    TexturePass(const TexturePass&) = delete;
    TexturePass(TexturePass&&) = delete;
    TexturePass& operator=(const TexturePass&) = delete;
    TexturePass& operator=(TexturePass&&) = delete;
    ~TexturePass();

    void process(unsigned int width, unsigned int height, GLuint inputTexture, GLuint outputTexture);

    private:
    GLuint framebuffer{};
    ScreenQuad screenQuad{};
    std::shared_ptr<resources::Shader> texturePassThrough{ nullptr };
};
}  // namespace spark