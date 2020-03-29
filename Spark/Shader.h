#ifndef SHADER_H
#define SHADER_H

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

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
    GLuint ID{0};
    std::map<GLenum, std::string> shaderSources{};

    std::set<Uniform> uniforms{};
    std::set<UniformBlock> uniformBlocks{};
    std::set<ShaderStorageBuffer> storageBuffers{};

    static inline std::string loadShader(const std::string& shaderPath);
    static inline std::map<GLenum, std::string> preProcess(const std::string& shaderPath);
    inline static GLenum shaderTypeFromString(const std::string& type);
    static inline std::vector<GLuint> compileShaders(const std::map<GLenum, std::string>& shaders);
    inline void linkProgram(const std::vector<GLuint>& ids);
    inline void acquireUniformNamesAndTypes();
    inline GLint getUniformLocation(const std::string& name) const;
    inline static std::string getUniformType(GLenum type);
    inline void acquireUniformBlocks();
    inline void acquireBuffers();
    inline std::optional<ShaderStorageBuffer> getShaderBuffer(const std::string& storageBufferName) const;
    inline std::optional<UniformBlock> getUniformBlock(const std::string& uniformBlockName) const;
};

}  // namespace spark
#endif