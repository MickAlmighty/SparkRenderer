#pragma once
#include <glad/glad.h>

namespace spark
{
class GBuffer
{
    public:
    GLuint framebuffer{};

    GLuint colorTexture{};
    GLuint normalsTexture{};
    GLuint roughnessMetalnessTexture{};
    GLuint depthTexture{};

    void setup(unsigned int width, unsigned int height);
    void cleanup();

    GBuffer() = default;
    GBuffer(const GBuffer& gBuffer) = delete;
    GBuffer(const GBuffer&& gBuffer) = delete;
    GBuffer operator=(const GBuffer& gBuffer) const = delete;
    GBuffer operator=(const GBuffer&& gBuffer) const = delete;

    ~GBuffer();

    private:
};
}  // namespace spark
