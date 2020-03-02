#ifndef SHADER_H
#define SHADER_H

#include <map>
#include <string>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <list>
#include <optional>
#include <set>
#include <vector>

namespace spark
{
struct Uniform;
struct UniformBlock;
struct ShaderStorageBuffer;
struct SSBO;
struct UniformBuffer;

class Shader
{
    public:
    std::string name;

    Shader(const std::string& vertexShaderPath, const std::string& fragmentShaderPath);
    Shader(const std::string& shaderPath);
    ~Shader();

    void use() const;

    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setVec2(const std::string& name, glm::vec2 value) const;
    void setVec3(const std::string& name, glm::vec3 value) const;
    void setMat4(const std::string& name, glm::mat4 value) const;
    void bindSSBO(const std::string& name, const SSBO& ssbo) const;
    void bindUniformBuffer(const std::string& name, const UniformBuffer& uniformBuffer) const;

    private:
    GLuint ID{0};
    std::set<Uniform> uniforms{};
    std::set<UniformBlock> uniformBlocks{};
    std::set<ShaderStorageBuffer> storageBuffers{};

    inline static GLenum shaderTypeFromString(const std::string& type);
    inline std::string loadShader(const std::string& shaderPath);
    inline std::map<GLenum, std::string> preProcess(const std::string& shaderPath);
    inline std::vector<GLuint> compileShaders(const std::map<GLenum, std::string>& shaders) const;
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