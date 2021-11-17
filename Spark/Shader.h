#pragma once

#include <map>
#include <vector>

#include <glm/glm.hpp>

#include "Buffer.hpp"
#include "Resource.h"
#include "ShaderDescriptor.hpp"

namespace spark::resources
{
class Shader : public resourceManagement::Resource
{
    public:
    Shader(const std::filesystem::path& path_);
    ~Shader() override;

    void use() const;
    void dispatchCompute(GLuint x, GLuint y, GLuint z) const;
    void dispatchComputeIndirect(GLuint bufferID) const;

    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setUInt(const std::string& name, unsigned int value) const;
    void setFloat(const std::string& name, float value) const;
    void setVec2(const std::string& name, glm::vec2 value) const;
    void setIVec2(const std::string& name, glm::ivec2 value) const;
    void setUVec2(const std::string& name, glm::uvec2 value) const;
    void setVec3(const std::string& name, glm::vec3 value) const;
    void setMat4(const std::string& name, glm::mat4 value) const;
    void bindSSBO(const std::string& name, const SSBO& ssbo);
    void bindUniformBuffer(const std::string& name, const UniformBuffer& uniformBuffer);

    private:
    std::vector<GLuint> compileShaders(const std::map<GLenum, std::string>& shaders);
    void linkProgram(const std::vector<GLuint>& ids);

    GLuint ID{0};
    ShaderDescriptor shaderDescriptor{};
};

}  // namespace spark::resources