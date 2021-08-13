#include "Shader.h"

#include <glm/gtc/type_ptr.hpp>

#include "Logging.h"
#include "ShaderParser.hpp"

namespace spark::resources
{
Shader::Shader(const std::filesystem::path& path_) : Resource(path_)
{
    const auto shaderIds = compileShaders(ShaderParser::parseShaderFile(path.string()));
    linkProgram(shaderIds);

    shaderDescriptor.acquireShaderResources(ID);
}

Shader::~Shader()
{
    glDeleteProgram(ID);
    ID = 0;
}

std::vector<GLuint> Shader::compileShaders(const std::map<GLenum, std::string>& shaders)
{
    std::vector<GLuint> shaderIds;
    shaderIds.reserve(shaders.size());
    for(const auto& [shaderType, shaderCode] : shaders)
    {
        const GLuint shader = glCreateShader(shaderType);
        const char* shaderSource = shaderCode.c_str();
        glShaderSource(shader, 1, &shaderSource, nullptr);
        glCompileShader(shader);
        GLint success;
        GLchar infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

        if(!success)
        {
            glGetShaderInfoLog(shader, 512, nullptr, infoLog);
            
            std::string fullInfo = getPath().string();
            fullInfo.append("\n\rERROR::SHADER::COMPILATION_FAILED, cause: ");
            fullInfo.append(infoLog);
            throw std::runtime_error(fullInfo);
        }
        shaderIds.push_back(shader);
    }

    return shaderIds;
}

void Shader::linkProgram(const std::vector<GLuint>& ids)
{
    const GLuint program = glCreateProgram();
    for(const auto id : ids)
    {
        glAttachShader(program, id);
    }

    glLinkProgram(program);

    GLint success;
    GLchar infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if(!success)
    {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        SPARK_ERROR("PROGRAM::LINKAGE_FAILED: {}", infoLog);
    }
    ID = program;

    for(const auto id : ids)
    {
        glDeleteShader(id);
    }
}

void Shader::use() const
{
    glUseProgram(ID);
}

void Shader::dispatchCompute(GLuint x, GLuint y, GLuint z) const
{
    glDispatchCompute(x, y, z);
}

void Shader::dispatchComputeIndirect(GLuint bufferID) const
{
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, bufferID);
    glDispatchComputeIndirect(0);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
}

void Shader::setBool(const std::string& name, bool value) const
{
    const GLint location = shaderDescriptor.getUniformLocation(name);
    if(location < 0)
        return;
    glUniform1i(location, value);
}

void Shader::setInt(const std::string& name, int value) const
{
    const GLint location = shaderDescriptor.getUniformLocation(name);
    if(location < 0)
        return;
    glUniform1i(location, value);
}

void Shader::setUInt(const std::string& name, unsigned int value) const
{
    const GLint location = shaderDescriptor.getUniformLocation(name);
    if(location < 0)
        return;
    glUniform1ui(location, value);
}

void Shader::setFloat(const std::string& name, float value) const
{
    const GLint location = shaderDescriptor.getUniformLocation(name);
    if(location < 0)
        return;
    glUniform1f(location, value);
}

void Shader::setVec2(const std::string& name, glm::vec2 value) const
{
    const GLint location = shaderDescriptor.getUniformLocation(name);
    if(location < 0)
        return;
    glUniform2fv(location, 1, glm::value_ptr(value));
}

void Shader::setIVec2(const std::string& name, glm::ivec2 value) const
{
    const GLint location = shaderDescriptor.getUniformLocation(name);
    if(location < 0)
        return;
    glUniform2iv(location, 1, glm::value_ptr(value));
}

void Shader::setUVec2(const std::string& name, glm::uvec2 value) const
{
    const GLint location = shaderDescriptor.getUniformLocation(name);
    if (location < 0)
        return;
    glUniform2uiv(location, 1, glm::value_ptr(value));
}

void Shader::setVec3(const std::string& name, glm::vec3 value) const
{
    const GLint location = shaderDescriptor.getUniformLocation(name);
    if(location < 0)
        return;
    glUniform3fv(location, 1, glm::value_ptr(value));
}

void Shader::setMat4(const std::string& name, glm::mat4 value) const
{
    const GLint location = shaderDescriptor.getUniformLocation(name);
    if(location < 0)
        return;
    glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(value));
}

void Shader::bindSSBO(const std::string& name, const SSBO& ssbo)
{
    const auto shaderBufferBlockIndex = shaderDescriptor.getShaderBufferBlockIndex(name);
    if(shaderBufferBlockIndex.has_value())
    {
        glShaderStorageBlockBinding(ID, *shaderBufferBlockIndex, ssbo.binding);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo.binding, ssbo.ID);
    }
}

void Shader::bindUniformBuffer(const std::string& name, const UniformBuffer& uniformBuffer)
{
    const auto uniformBlockIndex = shaderDescriptor.getUniformBlockIndex(name);
    if(uniformBlockIndex.has_value())
    {
        glUniformBlockBinding(ID, *uniformBlockIndex, uniformBuffer.binding);
        glBindBufferBase(GL_UNIFORM_BUFFER, uniformBuffer.binding, uniformBuffer.ID);
    }
}
}  // namespace spark::resources
