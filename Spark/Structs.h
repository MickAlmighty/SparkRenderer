#pragma once

#include <array>
#include <vector>
#include <filesystem>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <rttr/registration>

#include "LocalTransform.h"
#include "WorldTransform.h"

namespace spark
{
struct Cube;

struct Transform final
{
    LocalTransform local;
    WorldTransform world;
    RTTR_ENABLE();
};

namespace resources
{
    class Shader;
};

struct InitializationVariables final
{
    unsigned int width{};
    unsigned int height{};
    std::string pathToModels{};
    std::string pathToResources{};
    bool vsync{true};
    RTTR_ENABLE();
};

struct Texture
{
    Texture(GLuint id, const std::string& path);
    void setPath(const std::string path);
    std::string getPath() const;
    Texture() = default;
    GLuint ID{0};

    private:
    std::string path;
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE();
};

struct PbrCubemapTexture final
{
    GLuint cubemap{};
    GLuint irradianceCubemap{};
    GLuint prefilteredCubemap{};

    const std::string getPath();

    PbrCubemapTexture(GLuint hdrTexture, const std::string& path, unsigned int size = 1024);
    ~PbrCubemapTexture();

    static GLuint createIrradianceCubemap(GLuint framebuffer, GLuint environmentCubemap, Cube& cube,
                                          const std::shared_ptr<resources::Shader>& irradianceShader);
    static GLuint createPreFilteredCubemap(GLuint framebuffer, GLuint environmentCubemap, unsigned int envCubemapSize, Cube& cube,
                                           const std::shared_ptr<resources::Shader>& prefilterShader,
                                           const std::shared_ptr<resources::Shader>& resampleCubemapShader);

    private:
    std::string path{};

    void setup(GLuint hdrTexture, unsigned int cubemapSize);
    static GLuint createEnvironmentCubemapWithMipmapChain(GLuint framebuffer, GLuint equirectangularTexture, unsigned int size, Cube& cube,
                                                          const std::shared_ptr<resources::Shader>& equirectangularToCubemapShader);
    static GLuint createBrdfLookupTexture(GLuint framebuffer, unsigned int envCubemapSize, const std::shared_ptr<resources::Shader>& brdfShader);
    static GLuint createCubemapAndCopyDataFromFirstLayerOf(GLuint cubemap, unsigned int cubemapSize);
};

struct Vertex final
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoords;
    glm::vec3 tangent;
    glm::vec3 bitangent;
};

struct VertexShaderAttribute final
{
    unsigned int location{0};
    unsigned int components{1};  // 1 - 4
    unsigned int stride{};       // in bytes
    std::vector<uint8_t> bytes{};

    bool operator<(const VertexShaderAttribute& attribute) const
    {
        return location < attribute.location;
    }

    template<typename T>
    static VertexShaderAttribute createVertexShaderAttributeInfo(unsigned int location, unsigned int components, std::vector<T> vertexAttributeData)
    {
        unsigned int elemSize = sizeof(T);

        VertexShaderAttribute attribute;
        attribute.location = location;
        attribute.components = components;
        attribute.stride = elemSize;
        attribute.bytes.resize(elemSize * vertexAttributeData.size());
        std::memcpy(attribute.bytes.data(), vertexAttributeData.data(), vertexAttributeData.size() * elemSize);

        return attribute;
    }
};

struct QuadVertex final
{
    glm::vec3 pos;
    glm::vec2 texCoords;
};

struct ScreenQuad final
{
    GLuint vao{};
    GLuint vbo{};

    std::vector<QuadVertex> vertices = {
        {{-1.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}, {{1.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},  {{1.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},

        {{-1.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}, {{-1.0f, -1.0f, 0.0f}, {0.0f, 0.0f}}, {{1.0f, -1.0f, 0.0f}, {1.0f, 0.0f}}};

    void setup()
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(QuadVertex), &vertices[0], GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), reinterpret_cast<void*>(offsetof(QuadVertex, pos)));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), reinterpret_cast<void*>(offsetof(QuadVertex, texCoords)));

        glBindVertexArray(0);
    }

    void draw() const
    {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
        glBindVertexArray(0);
    }

    ~ScreenQuad()
    {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
};

struct Cube final
{
    GLuint vao{};
    GLuint vbo{};

    Cube();

    void draw() const
    {
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
    }

    ~Cube()
    {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
};

struct DirectionalLightData final
{
    alignas(16) glm::vec3 direction;
    alignas(16) glm::vec3 color;  // strength baked into color
};

struct PointLightData final
{
    alignas(16) glm::vec4 positionAndRadius;
    alignas(16) glm::vec3 color;  // strength baked into color
    alignas(16) glm::mat4 modelMat;
};

struct SpotLightData final
{
    alignas(16) glm::vec3 position;
    float cutOff;
    glm::vec3 color;  // strength baked into color
    float outerCutOff;
    glm::vec3 direction;
    float maxDistance;
    glm::vec4 boundingSphere;  // for cone culling approximation
};

struct LightProbeData final
{
    GLint64 irradianceCubemapHandle{0};
    GLint64 prefilterCubemapHandle{0};
    alignas(16) glm::vec3 position{0};
    float radius{0};
    float fadeDistance{0};
};

struct DrawArraysIndirectCommand final
{
    GLuint count;
    GLuint instanceCount;
    GLuint first;
    GLuint baseInstance;
};

}  // namespace spark