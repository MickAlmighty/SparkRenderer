#pragma once

#include <string>
#include <vector>

#include "glad_glfw3.h"

namespace spark::utils
{
class ShaderCompiler
{
    public:
    static std::vector<unsigned> compileVulkan(const std::string& source_name, GLenum glShaderType, const std::string& source, bool optimize = false);
    static std::vector<unsigned> compileOpenGL(const std::string& source_name, GLenum glShaderType, const std::string& source, bool optimize = false);

    private:
    ~ShaderCompiler() = default;
};
}  // namespace spark::utils