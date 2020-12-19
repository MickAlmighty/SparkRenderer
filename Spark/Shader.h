#ifndef SHADER_H
#define SHADER_H

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

#include "Buffer.hpp"
#include "GPUResource.h"
#include "Resource.h"
#include "Structs.h"

namespace spark::resources
{
class Shader : public resourceManagement::Resource, public resourceManagement::GPUResource
{
    public:
    Shader(const resourceManagement::ResourceIdentifier& identifier);
    ~Shader();

    bool isResourceReady() override;
    bool gpuLoad() override;
    bool gpuUnload() override;
    bool load() override;
    bool unload() override;

    void use() const;
    void dispatchCompute(GLuint x, GLuint y, GLuint z) const;

    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setUInt(const std::string& name, unsigned int value) const;
    void setFloat(const std::string& name, float value) const;
    void setVec2(const std::string& name, glm::vec2 value) const;
    void setIVec2(const std::string& name, glm::ivec2 value) const;
    void setVec3(const std::string& name, glm::vec3 value) const;
    void setMat4(const std::string& name, glm::mat4 value) const;
    void bindSSBO(const std::string& name, const SSBO& ssbo) const;
    void bindUniformBuffer(const std::string& name, const UniformBuffer& uniformBuffer) const;

    private:
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

    GLuint ID{0};
    std::map<GLenum, std::string> shaderSources{};

    std::set<Uniform> uniforms{};
    std::set<UniformBlock> uniformBlocks{};
    std::set<ShaderStorageBuffer> storageBuffers{};

    static inline std::string loadShader(const std::string& shaderPath);
    static inline std::map<GLenum, std::string> preProcess(const std::string& shaderSource);
    static inline GLenum shaderTypeFromString(const std::string& type);
    static inline std::vector<GLuint> compileShaders(const std::map<GLenum, std::string>& shaders);
    void linkProgram(const std::vector<GLuint>& ids);
    void acquireUniformNamesAndTypes();
    GLint getUniformLocation(const std::string& name) const;
    static std::string getUniformType(GLenum type);
    void acquireUniformBlocks();
    void acquireBuffers();
    std::optional<ShaderStorageBuffer> getShaderBuffer(const std::string& storageBufferName) const;
    std::optional<UniformBlock> getUniformBlock(const std::string& uniformBlockName) const;
};

}  // namespace spark::resources
#endif