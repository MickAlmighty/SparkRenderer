#pragma once

#include <string>
#include <vector>

#include "glad_glfw3.h"

namespace spark::utils
{
class ShaderCompiler
{
    public:
    static std::vector<unsigned> compile(const std::string& source_name, GLenum glShaderType, const std::string& source, bool optimize = false);

    private:
    ~ShaderCompiler() = default;
};
}  // namespace spark::utils