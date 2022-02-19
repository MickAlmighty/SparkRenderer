#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <queue>

#include "glad_glfw3.h"

namespace spark::utils
{
class ShaderParser
{
    public:
    static std::map<GLenum, std::string> splitShaderSourceByStages(const std::string& shaderPath);
    static bool any_of_files_from_inlude_chain_modified(const std::filesystem::path& shaderPath,
                                                        const std::filesystem::path& cachedShaderAbsolutePath);

    private:
    ~ShaderParser() = default;

    static std::map<GLenum, std::string> preProcess(const std::string& shaderSource);
    static GLenum shaderTypeFromString(const std::string& type);

    static std::vector<std::filesystem::path> ShaderParser::get_file_dependency_chain(const std::filesystem::path& file);
    static void look_for_includes_in_file(const std::filesystem::path& file, std::queue<std::filesystem::path>& files_to_traverse);
    static std::string::size_type find_path_beginning_sign_idx(const std::string& includeLine, bool& isPathAbsolute);
};
}  // namespace spark::utils