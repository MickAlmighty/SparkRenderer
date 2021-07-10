#pragma once
#include "ShaderParser.hpp"

#include <sstream>
#include <fstream>

#include "Logging.h"

namespace spark
{
std::map<GLenum, std::string> ShaderParser::parseShaderFile(const std::string& shaderPath)
{
    const auto shaderSource = loadShaderStringFromFile(shaderPath);
    return preProcess(shaderSource);
}

std::string ShaderParser::loadShaderStringFromFile(const std::string& shaderPath)
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
    catch (const std::ifstream::failure& e)
    {
        SPARK_ERROR("SHADER::FILE_NOT_SUCCESSFULLY_READ: {}, {}", e.what(), shaderPath);
    }
    return codeString;
}

std::map<GLenum, std::string> ShaderParser::preProcess(const std::string& shaderSource)
{
    std::map<GLenum, std::string> shaderSources;

    const char* typeToken = "#type";
    const size_t typeTokenLength = strlen(typeToken);
    size_t pos = shaderSource.find(typeToken, 0);
    while (pos != std::string::npos)
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

GLenum ShaderParser::shaderTypeFromString(const std::string& type)
{
    if (type == "vertex")
    {
        return GL_VERTEX_SHADER;
    }
    if (type == "fragment" || type == "pixel")
    {
        return GL_FRAGMENT_SHADER;
    }
    if (type == "geometry")
    {
        return GL_GEOMETRY_SHADER;
    }
    if (type == "compute")
    {
        return GL_COMPUTE_SHADER;
    }

    return 0;
}
}  // namespace spark