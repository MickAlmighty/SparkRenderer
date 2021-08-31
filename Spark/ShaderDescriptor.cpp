#include "ShaderDescriptor.hpp"

#include "Logging.h"

namespace spark
{
void ShaderDescriptor::acquireShaderResources(const GLuint ID)
{
    uniforms.clear();
    uniformBlocks.clear();
    storageBuffers.clear();

    acquireUniformNamesAndTypes(ID);
    acquireUniformBlocks(ID);
    acquireBuffers(ID);
}

GLint ShaderDescriptor::getUniformLocation(const std::string& name) const
{
    const auto uniform_it = std::find_if(std::begin(uniforms), std::end(uniforms), [&name](const Uniform& uniform) { return uniform.name == name; });

    if(uniform_it != std::end(uniforms))
    {
        return static_cast<GLuint>(uniform_it->location);
    }
    return -1;
}

std::optional<std::size_t> ShaderDescriptor::getShaderBufferBlockIndex(const std::string& storageBufferName) const
{
    const auto storageBufferIt = std::find_if(storageBuffers.begin(), storageBuffers.end(),
                                              [&storageBufferName](const ShaderStorageBuffer& buffer) { return buffer.name == storageBufferName; });

    if(storageBufferIt != storageBuffers.end())
    {
        return storageBufferIt->blockIndex;
    }

    return std::nullopt;
}

std::optional<std::size_t> ShaderDescriptor::getUniformBlockIndex(const std::string& uniformBlockName) const
{
    const auto uniformBlockIt = std::find_if(uniformBlocks.begin(), uniformBlocks.end(),
                                             [&uniformBlockName](const UniformBlock& buffer) { return buffer.name == uniformBlockName; });

    if(uniformBlockIt != uniformBlocks.end())
    {
        return uniformBlockIt->blockIndex;
    }

    return std::nullopt;
}

void ShaderDescriptor::acquireUniformNamesAndTypes(const GLuint ID)
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

void ShaderDescriptor::acquireUniformBlocks(const GLuint ID)
{
    GLint numberOfUniformBlocks{-1};
    glGetProgramInterfaceiv(ID, GL_UNIFORM_BLOCK, GL_ACTIVE_RESOURCES, &numberOfUniformBlocks);

    for(int index = 0; index < numberOfUniformBlocks; ++index)
    {
        std::string uniformBlockName(256, '0');
        GLsizei size = 0;
        glGetProgramResourceName(ID, GL_UNIFORM_BLOCK, index, static_cast<GLsizei>(uniformBlockName.size()), &size, uniformBlockName.data());
        uniformBlockName = uniformBlockName.substr(0, size);
        const GLint uniformBlockIndex = glGetProgramResourceIndex(ID, GL_UNIFORM_BLOCK, uniformBlockName.c_str());

        uniformBlocks.insert({uniformBlockName, uniformBlockIndex});
    }
}

void ShaderDescriptor::acquireBuffers(const GLuint ID)
{
    GLint numberOfShaderBuffers{-1};
    glGetProgramInterfaceiv(ID, GL_SHADER_STORAGE_BLOCK, GL_ACTIVE_RESOURCES, &numberOfShaderBuffers);

    for(int index = 0; index < numberOfShaderBuffers; ++index)
    {
        std::string shaderStorageBufferName(256, '0');
        GLsizei size = 0;
        glGetProgramResourceName(ID, GL_SHADER_STORAGE_BLOCK, index, static_cast<GLsizei>(shaderStorageBufferName.size()), &size,
                                 shaderStorageBufferName.data());
        shaderStorageBufferName = shaderStorageBufferName.substr(0, size);
        const GLint uniformBlockIndex = glGetProgramResourceIndex(ID, GL_SHADER_STORAGE_BLOCK, shaderStorageBufferName.c_str());

        storageBuffers.insert({shaderStorageBufferName, uniformBlockIndex});
    }
}

std::string ShaderDescriptor::getUniformType(GLenum type)
{
    switch(type)
    {
        case GL_FLOAT:
            return "float";
        case GL_UNSIGNED_INT:
            return "unsigned int";
        case GL_INT:
            return "int";
        case GL_BOOL:
            return "bool";
        case GL_INT_VEC2:
            return "ivec2";
        case GL_UNSIGNED_INT_VEC2:
            return "uvec2";
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
        case GL_IMAGE_1D:
            return "image1D";
        case GL_IMAGE_2D:
            return "image2D";
        case GL_IMAGE_3D:
            return "image3D";
        default:
        {
            SPARK_INFO("Unknown uniform type! 0x{0:x}", type);
            return "";
        }
    }
}
}  // namespace spark