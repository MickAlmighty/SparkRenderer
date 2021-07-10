#pragma once

#include <map>
#include <string>

#include "glad_glfw3.h"

namespace spark
{
class ShaderParser
{
    public:
    static std::map<GLenum, std::string> parseShaderFile(const std::string& shaderPath);

    private:
    ~ShaderParser() = default;

    static std::string loadShaderStringFromFile(const std::string& shaderPath);
    static std::map<GLenum, std::string> preProcess(const std::string& shaderSource);
    static GLenum shaderTypeFromString(const std::string& type);
};
}  // namespace spark