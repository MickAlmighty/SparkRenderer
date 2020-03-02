﻿#include "Shader.h"

#include <iostream>
#include <fstream>
#include <numeric>
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
    acquireUniformBlocks();
    acquireBuffers();
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
    acquireUniformBlocks();
    acquireBuffers();
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
    int32_t numberOfUniforms{-1};
    glGetProgramInterfaceiv(ID, GL_UNIFORM, GL_ACTIVE_RESOURCES, &numberOfUniforms);
    if(numberOfUniforms < 1)
        return;

    GLenum properties[4] = {GL_BLOCK_INDEX, GL_TYPE, GL_NAME_LENGTH, GL_LOCATION};

    for(int i = 0; i < numberOfUniforms; ++i)
    {
        GLint values[4];
        glGetProgramResourceiv(ID, GL_UNIFORM, i, 4, properties, 4, nullptr, values);

        // Skip any uniforms that are in a block.
        if(values[0] != -1)
            continue;

        std::string uniformName;
        uniformName.resize(values[2]);
        GLsizei size = 0;

        glGetActiveUniformName(ID, i, static_cast<GLsizei>(uniformName.size()), &size, uniformName.data());
        uniformName = uniformName.substr(0, uniformName.size() - 1);

        uniforms.insert({uniformName, getUniformType(values[1]), values[3]});
    }
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
        default:
            return "";
    }
}

void Shader::acquireUniformBlocks()
{
    GLint numberOfUniformBlocks{-1};
    glGetProgramInterfaceiv(ID, GL_UNIFORM_BLOCK, GL_ACTIVE_RESOURCES, &numberOfUniformBlocks);

    for(int index = 0; index < numberOfUniformBlocks; ++index)
    {
        std::string uniformBlockName;
        uniformBlockName.resize(100);
        GLsizei size = 0;
        glGetProgramResourceName(ID, GL_UNIFORM_BLOCK, index, static_cast<GLsizei>(uniformBlockName.size()), &size, uniformBlockName.data());
        uniformBlockName = uniformBlockName.substr(0, size);

        const GLint uniformBlockIndex = glGetProgramResourceIndex(ID, GL_UNIFORM_BLOCK, uniformBlockName.c_str());

        uniformBlocks.insert({uniformBlockName, uniformBlockIndex});
    }
}

void Shader::acquireBuffers()
{
    GLint numberOfShaderBuffers{-1};
    glGetProgramInterfaceiv(ID, GL_SHADER_STORAGE_BLOCK, GL_ACTIVE_RESOURCES, &numberOfShaderBuffers);

    for(int index = 0; index < numberOfShaderBuffers; ++index)
    {
        std::string shaderStorageBufferName;
        shaderStorageBufferName.resize(100);
        GLsizei size = 0;
        glGetProgramResourceName(ID, GL_SHADER_STORAGE_BLOCK, index, static_cast<GLsizei>(shaderStorageBufferName.size()), &size,
                                 shaderStorageBufferName.data());
        shaderStorageBufferName = shaderStorageBufferName.substr(0, size);

        const GLint uniformBlockIndex = glGetProgramResourceIndex(ID, GL_SHADER_STORAGE_BLOCK, shaderStorageBufferName.c_str());

        storageBuffers.insert({shaderStorageBufferName, uniformBlockIndex});
    }
}

std::optional<ShaderStorageBuffer> Shader::getShaderBuffer(const std::string& storageBufferName) const
{
    const auto storageBufferIt = std::find_if(storageBuffers.begin(), storageBuffers.end(),
                                              [&storageBufferName](const ShaderStorageBuffer& buffer) { return buffer.name == storageBufferName; });

    if(storageBufferIt != storageBuffers.end())
    {
        return *storageBufferIt;
    }

    return std::nullopt;
}

std::optional<UniformBlock> Shader::getUniformBlock(const std::string& uniformBlockName) const
{
    const auto uniformBlockIt = std::find_if(uniformBlocks.begin(), uniformBlocks.end(),
                                             [&uniformBlockName](const UniformBlock& buffer) { return buffer.name == uniformBlockName; });

    if(uniformBlockIt != uniformBlocks.end())
    {
        return *uniformBlockIt;
    }

    return std::nullopt;
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

GLint Shader::getUniformLocation(const std::string& name) const
{
    const auto uniform_it = std::find_if(std::begin(uniforms), std::end(uniforms), [&name](const Uniform& uniform) { return uniform.name == name; });

    if(uniform_it != std::end(uniforms))
    {
        return static_cast<GLuint>(uniform_it->location);
    }
    return -1;
}

void Shader::setBool(const std::string& name, bool value) const
{
    const GLint location = getUniformLocation(name);
    if(location < 0)
        return;
    glUniform1i(getUniformLocation(name), value);
}

void Shader::setInt(const std::string& name, int value) const
{
    const GLint location = getUniformLocation(name);
    if(location < 0)
        return;
    glUniform1i(getUniformLocation(name), value);
}

void Shader::setFloat(const std::string& name, float value) const
{
    const GLint location = getUniformLocation(name);
    if(location < 0)
        return;
    glUniform1f(getUniformLocation(name), value);
}

void Shader::setVec2(const std::string& name, glm::vec2 value) const
{
    const GLint location = getUniformLocation(name);
    if(location < 0)
        return;
    glUniform2fv(location, 1, glm::value_ptr(value));
}

void Shader::setVec3(const std::string& name, glm::vec3 value) const
{
    const GLint location = getUniformLocation(name);
    if(location < 0)
        return;
    glUniform3fv(getUniformLocation(name), 1, glm::value_ptr(value));
}

void Shader::setMat4(const std::string& name, glm::mat4 value) const
{
    const GLint location = getUniformLocation(name);
    if(location < 0)
        return;
    glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(value));
}

void Shader::bindSSBO(const std::string& name, const SSBO& ssbo) const
{
    const auto shaderBuffer = getShaderBuffer(name);
    if(shaderBuffer.has_value())
    {
        glShaderStorageBlockBinding(ID, shaderBuffer->blockIndex, ssbo.binding);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo.binding, ssbo.ID);
    }
}

void Shader::bindUniformBuffer(const std::string& name, const UniformBuffer& uniformBuffer) const
{
    const auto uniformBlock = getUniformBlock(name);
    if(uniformBlock.has_value())
    {
        glUniformBlockBinding(ID, uniformBlock->blockIndex, uniformBuffer.binding);
        glBindBufferBase(GL_UNIFORM_BUFFER, uniformBuffer.binding, uniformBuffer.ID);
    }
}
}  // namespace spark
