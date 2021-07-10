#pragma once

#include <string>
#include <map>
#include <optional>
#include <set>

#include "glad_glfw3.h"

namespace spark
{
class ShaderDescriptor
{
    public:
    void acquireShaderResources(const GLuint ID);
    GLint getUniformLocation(const std::string& name) const;
    std::optional<std::size_t> getShaderBufferBlockIndex(const std::string& storageBufferName) const;
    std::optional<std::size_t> getUniformBlockIndex(const std::string& uniformBlockName) const;

    private:
    void acquireUniformNamesAndTypes(const GLuint ID);
    void acquireUniformBlocks(const GLuint ID);
    void acquireBuffers(const GLuint ID);
    static std::string getUniformType(GLenum type);


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

    std::set<Uniform> uniforms{};
    std::set<UniformBlock> uniformBlocks{};
    std::set<ShaderStorageBuffer> storageBuffers{};
};

}  // namespace spark