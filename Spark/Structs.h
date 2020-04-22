#ifndef STRUCTS_H
#define STRUCTS_H

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

struct Uniform final
{
    std::string name{};
    std::string type{};
    GLint location{0};

    bool operator!=(const Uniform& rhs) const
    {
        return this->name != rhs.name || this->type != rhs.type;
    }
    bool operator<(const Uniform& rhs) const
    {
        return this->name < rhs.name;
    }
};

struct UniformBlock final
{
    std::string name{};
    GLint blockIndex{0};

    bool operator!=(const UniformBlock& rhs) const
    {
        return this->name != rhs.name;
    }
    bool operator<(const UniformBlock& rhs) const
    {
        return this->name < rhs.name;
    }
};

struct ShaderStorageBuffer final
{
    std::string name{};
    GLint blockIndex{0};

    bool operator!=(const ShaderStorageBuffer& rhs) const
    {
        return this->name != rhs.name;
    }
    bool operator<(const ShaderStorageBuffer& rhs) const
    {
        return this->name < rhs.name;
    }
};

struct InitializationVariables final
{
    unsigned int width;
    unsigned int height;
    std::string pathToModels;
    std::string pathToResources;
    bool vsync;
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
    GLuint brdfLUTTexture{};

    const std::string getPath();

    PbrCubemapTexture(GLuint hdrTexture, const std::string& path, unsigned int size = 1024);
    ~PbrCubemapTexture();

    static GLuint createIrradianceCubemap(GLuint framebuffer, GLuint environmentCubemap, Cube& cube, glm::mat4 projection,
                                   const std::array<glm::mat4, 6>& views, const std::shared_ptr<resources::Shader>& irradianceShader);
    static GLuint createPreFilteredCubemap(GLuint framebuffer, GLuint environmentCubemap, unsigned int envCubemapSize, Cube& cube, glm::mat4 projection,
                                   const std::array<glm::mat4, 6>& views, const std::shared_ptr<resources::Shader>& prefilterShader);

    private:
    std::string path{};
    std::array<glm::mat4, 6> viewMatrices{};
    glm::mat4 projection{};

    std::shared_ptr<resources::Shader> equirectangularToCubemapShader{nullptr};
    std::shared_ptr<resources::Shader> irradianceShader{nullptr};
    std::shared_ptr<resources::Shader> prefilterShader{nullptr};
    std::shared_ptr<resources::Shader> brdfShader{nullptr};

    void setup(GLuint hdrTexture, unsigned int size);
    GLuint createEnvironmentCubemapWithMipmapChain(GLuint framebuffer, GLuint equirectangularTexture, unsigned int size, Cube& cube) const;
    GLuint createBrdfLookupTexture(GLuint framebuffer, unsigned int envCubemapSize) const;
    GLuint createCubemapAndCopyDataFromFirstLayerOf(GLuint cubemap, unsigned int cubemapSize) const;
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

template<GLenum BUFFER_TYPE>
struct Buffer
{
    static inline std::set<uint32_t> bindings{};
    static inline std::set<uint32_t> freedBindings{};

    GLuint ID{0};
    GLint binding{-1};
    GLsizei size{0};

    Buffer() = default;
    Buffer(const Buffer& buffer) = default;
    Buffer& operator=(const Buffer& buffer) = default;
    ~Buffer() = default;

    void genBuffer(size_t sizeInBytes = 0)
    {
        size = static_cast<GLsizei>(sizeInBytes);
        glGenBuffers(1, &ID);
        glBindBuffer(BUFFER_TYPE, ID);
        glBufferData(BUFFER_TYPE, size, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(BUFFER_TYPE, 0);
        getBinding();
    }

    void bind() const
    {
        glBindBuffer(BUFFER_TYPE, ID);
    }

    void unbind() const
    {
        glBindBuffer(BUFFER_TYPE, 0);
    }

    template<typename T>
    void updateData(const std::vector<T>& buffer)
    {
        const size_t vectorSize = buffer.size() * sizeof(T);
        if(vectorSize < size || vectorSize > size)
        {
            // SPARK_WARN("Trying to update SSBO with a vector with too large size! SSBO size: {}, vector size: {}. Buffer will be resized and update
            // will be processed!", size, vectorSize);
            size = static_cast<GLsizei>(vectorSize);
            glNamedBufferData(ID, vectorSize, buffer.data(), GL_DYNAMIC_DRAW);
        }
        else
        {
            glNamedBufferSubData(ID, 0, vectorSize, buffer.data());
        }
    }

    template<typename T>
    void updateSubData(size_t offsetFromBeginning, const std::vector<T>& buffer)
    {
        const size_t vectorSize = buffer.size() * sizeof(T);
        const GLintptr offset = static_cast<GLintptr>(offsetFromBeginning);

        if(offset > size)
        {
            return;
        }
        if(offset + vectorSize > size)
        {
            return;
        }

        glNamedBufferSubData(ID, offset, vectorSize, buffer.data());
    }

    void clearBuffer() const
    {
        glClearNamedBufferData(ID, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    }

    void cleanup()
    {
        glDeleteBuffers(1, &ID);
        ID = 0;
        freeBinding();
    }

    private:
    void getBinding()
    {
        if(!freedBindings.empty())
        {
            binding = *freedBindings.begin();
            freedBindings.erase(freedBindings.begin());
            return;
        }
        if(bindings.empty())
        {
            bindings.insert(0);
            binding = 0;
        }
        else
        {
            binding = *std::prev(bindings.end()) + 1;
            bindings.insert(binding);
        }
    };

    void freeBinding()
    {
        const auto it = bindings.find(binding);
        if(it != bindings.end())
        {
            freedBindings.insert(*it);
            bindings.erase(it);
            binding = -1;
        }
    }
};
using SSBO = Buffer<GL_SHADER_STORAGE_BUFFER>;
using UniformBuffer = Buffer<GL_UNIFORM_BUFFER>;
using ElementArrayBuffer = Buffer<GL_ELEMENT_ARRAY_BUFFER>;
using VertexBuffer = Buffer<GL_ARRAY_BUFFER>;

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
};

struct DrawArraysIndirectCommand final
{
    GLuint count;
    GLuint instanceCount;
    GLuint first;
    GLuint baseInstance;
};

}  // namespace spark
#endif