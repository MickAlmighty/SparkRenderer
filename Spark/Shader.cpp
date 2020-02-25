#include "Shader.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <glm/gtc/type_ptr.hpp>

#include "Structs.h"
#include "Logging.h"

namespace spark
{
GLenum Shader::shaderTypeFromString(const std::string& type)
{
    if(type == "vertex")
    {
        return GL_VERTEX_SHADER;
    }
    if(type == "fragment" || type == "pixel")
    {
        return GL_FRAGMENT_SHADER;
    }
    if(type == "geometry")
    {
        return GL_GEOMETRY_SHADER;
    }

    return 0;
}

Shader::Shader(const std::string& vertexShaderPath, const std::string& fragmentShaderPath)
{
    const auto beginNameIndex = vertexShaderPath.find_last_of('\\') + 1;
    const auto endNameIndex = vertexShaderPath.find_last_of('.');
    const std::string shaderName = vertexShaderPath.substr(beginNameIndex, endNameIndex - beginNameIndex);
    name = shaderName;

    const std::string vertexCode = loadShader(vertexShaderPath);
    const std::string fragmentCode = loadShader(fragmentShaderPath);

    std::map<GLenum, std::string> shaders;
    shaders.emplace(GL_VERTEX_SHADER, vertexCode);
    shaders.emplace(GL_FRAGMENT_SHADER, fragmentCode);

    const auto shaderIds = compileShaders(shaders);
    linkProgram(shaderIds);

    acquireUniformNamesAndTypes();
}

Shader::Shader(const std::string& shaderPath)
{
    const std::string shaderSource = loadShader(shaderPath);
    const auto shaders = preProcess(shaderSource);

    const auto beginNameIndex = shaderPath.find_last_of('\\') + 1;
    const auto endNameIndex = shaderPath.size();
    const std::string shaderName = shaderPath.substr(beginNameIndex, endNameIndex - beginNameIndex);
    name = shaderName;

    const auto shaderIds = compileShaders(shaders);
    linkProgram(shaderIds);

    acquireUniformNamesAndTypes();
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

void Shader::acquireUniformNamesAndTypes()
{
    int32_t numberOfUniforms = -1;
    glGetProgramiv(ID, GL_ACTIVE_UNIFORMS, &numberOfUniforms);
    if(numberOfUniforms < 1)
        return;

    std::vector<GLuint> uniformIndices(numberOfUniforms);
    std::vector<GLint> uniformNamesLength(numberOfUniforms);
    std::vector<GLenum> uniformTypes(numberOfUniforms);

    for(int32_t uniformIndex = 0; uniformIndex < numberOfUniforms; ++uniformIndex)
    {
        uniformIndices[uniformIndex] = uniformIndex;
    }

    glGetActiveUniformsiv(ID, static_cast<GLsizei>(uniformIndices.size()), uniformIndices.data(), GL_UNIFORM_NAME_LENGTH, uniformNamesLength.data());

    glGetActiveUniformsiv(ID, static_cast<GLsizei>(uniformIndices.size()), uniformIndices.data(), GL_UNIFORM_TYPE,
                          reinterpret_cast<GLint*>(uniformTypes.data()));

    for(int i = 0; i < numberOfUniforms; ++i)
    {
        std::string uniformName;
        uniformName.resize(uniformNamesLength[i]);
        GLsizei size = 0;

        glGetActiveUniformName(ID, uniformIndices[i], static_cast<GLsizei>(uniformName.size()), &size, uniformName.data());
        uniformName = uniformName.substr(0, uniformName.size() - 1);

        uniforms.insert({uniformName, getUniformType(uniformTypes[i]), getUniformLocation(uniformName)});
    }
}

GLint Shader::getUniformLocation(const std::string& uniformName) const
{
    return glGetUniformLocation(ID, uniformName.c_str());
}

std::string Shader::getUniformType(GLenum type)
{
    switch(type)
    {
        case GL_FLOAT:
            return "float";
        case GL_INT:
            return "int";
        case GL_BOOL:
            return "bool";
        case GL_FLOAT_VEC2:
            return "vec2";
        case GL_FLOAT_VEC3:
            return "vec3";
        case GL_FLOAT_VEC4:
            return "vec4";
        case GL_FLOAT_MAT2:
            return "mat2";
        case GL_FLOAT_MAT3:
            return "mat3";
        case GL_FLOAT_MAT4:
            return "mat4";
        case GL_SAMPLER_1D:
            return "sampler1D";
        case GL_SAMPLER_1D_ARRAY:
            return "sampler1D_array";
        case GL_SAMPLER_2D:
            return "sampler2D";
        case GL_SAMPLER_2D_ARRAY:
            return "sampler2D_array";
        case GL_SAMPLER_CUBE:
            return "samplerCube";
        default: return "";
    }
}

Shader::~Shader()
{
    glDeleteProgram(ID);
    SPARK_TRACE("Shader deleted!");
}

std::string Shader::loadShader(const std::string& shaderPath)
{
    std::string codeString;
    std::stringstream shaderStream;
    try
    {
        std::ifstream shaderFile;
        shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        shaderFile.open(shaderPath);
        shaderStream << shaderFile.rdbuf();
        shaderFile.close();

        codeString = shaderStream.str();
    }
    catch(const std::ifstream::failure& e)
    {
        SPARK_ERROR("SHADER::FILE_NOT_SUCCESSFULLY_READ: {}", e.what());
    }
    return codeString;
}

std::map<GLenum, std::string> Shader::preProcess(const std::string& shaderSource)
{
    std::map<GLenum, std::string> shaderSources;

    const char* typeToken = "#type";
    const size_t typeTokenLength = strlen(typeToken);
    size_t pos = shaderSource.find(typeToken, 0);
    while(pos != std::string::npos)
    {
        const size_t eol = shaderSource.find_first_of("\r\n", pos);
        const size_t begin = pos + typeTokenLength + 1;
        std::string type = shaderSource.substr(begin, eol - begin);

        const size_t nextLinePos = shaderSource.find_first_not_of("\r\n", eol);
        pos = shaderSource.find(typeToken, nextLinePos);
        shaderSources[shaderTypeFromString(type)] =
            shaderSource.substr(nextLinePos, pos - (nextLinePos == std::string::npos ? shaderSource.size() - 1 : nextLinePos));
    }

    return shaderSources;
}

std::vector<GLuint> Shader::compileShaders(const std::map<GLenum, std::string>& shaders) const
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
            std::runtime_error("ERROR::SHADER::COMPILATION_FAILED/n");
        }
        shaderIds.push_back(shader);
    }

    return shaderIds;
}

void Shader::use() const
{
    glUseProgram(ID);
}

GLuint Shader::getLocation(const std::string& name) const
{
    const auto uniform_it =
        std::find_if(std::begin(uniforms), std::end(uniforms), [&name](const Uniform& uniform)
    {
        return uniform.name == name;
    });

    if(uniform_it != std::end(uniforms))
    {
        return static_cast<GLuint>(uniform_it->location);
    }
    return 0;
}

GLuint Shader::getBufferBinding(const std::string& name) const
{
    const auto bufferBindingIt = bufferBindings.find(name);
    if(bufferBindingIt != std::end(bufferBindings))
    {
        return bufferBindingIt->second;
    }

    if(bufferBindings.empty())
    {
        bufferBindings[name] = 0;
        return bufferBindings[name];
    }
    else
    {
        auto end = bufferBindings.end();
        auto lastIt = std::prev(end);
        bufferBindings[name] = lastIt->second + 1;
        return bufferBindings[name];
    }
}

void Shader::setBool(const std::string& name, bool value) const
{
    glUniform1i(getLocation(name), value);
}

void Shader::setInt(const std::string& name, int value) const
{
    glUniform1i(getLocation(name), value);
}

void Shader::setFloat(const std::string& name, float value) const
{
    glUniform1f(getLocation(name), value);
}

void Shader::setVec2(const std::string& name, glm::vec2 value) const
{
    glUniform2fv(getLocation(name), 1, glm::value_ptr(value));
}

void Shader::setVec3(const std::string& name, glm::vec3 value) const
{
    glUniform3fv(getLocation(name), 1, glm::value_ptr(value));
}

void Shader::setMat4(const std::string& name, glm::mat4 value) const
{
    glUniformMatrix4fv(getLocation(name), 1, GL_FALSE, glm::value_ptr(value));
}

void Shader::bindSSBO(const std::string& name, GLuint ssbo) const
{
    const GLuint block_index = glGetProgramResourceIndex(ID, GL_SHADER_STORAGE_BLOCK, name.c_str());
    const GLuint binding = getBufferBinding(name);
    glShaderStorageBlockBinding(ID, block_index, binding);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, ssbo);
}

}  // namespace spark
