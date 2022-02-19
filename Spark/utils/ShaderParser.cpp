#pragma once
#include "ShaderParser.hpp"

#include <sstream>
#include <fstream>
#include <regex>

#include "Logging.h"
#include "utils/FileUtils.hpp"

namespace
{
const std::regex regex(R"((#include[\s]*["][\S]+["])|(#include[\s]*[<][\S]+[>]))");
}

namespace spark::utils
{
std::map<GLenum, std::string> ShaderParser::splitShaderSourceByStages(const std::string& shaderPath)
{
    const auto shaderSource = utils::load_text_from_file(shaderPath);
    return preProcess(shaderSource);
}

bool ShaderParser::any_of_files_from_inlude_chain_modified(const std::filesystem::path& shaderPath,
                                                           const std::filesystem::path& cachedShaderAbsolutePath)
{
    const auto cachedShaderFileModificationTime = std::filesystem::last_write_time(cachedShaderAbsolutePath).time_since_epoch().count();
    const auto includes = get_file_dependency_chain(shaderPath);

    for(const auto& include : includes)
    {
        const auto includePath = std::filesystem::path(include);
        const auto includeFileModificationTime = std::filesystem::last_write_time(includePath).time_since_epoch().count();

        if(auto isCacheFileOlder = cachedShaderFileModificationTime < includeFileModificationTime; isCacheFileOlder)
        {
            return true;
        }
    }

    return false;
}

std::vector<std::filesystem::path> ShaderParser::get_file_dependency_chain(const std::filesystem::path& file)
{
    std::queue<std::filesystem::path> files_to_traverse;
    std::vector<std::filesystem::path> files_traversed;

    files_to_traverse.push(file);

    while(!files_to_traverse.empty())
    {
        const std::filesystem::path& file_to_traverse = files_to_traverse.front();

        const auto find_if_traversed = [&file_to_traverse](const std::filesystem::path& path) { return file_to_traverse == path; };
        const auto it = std::find_if(files_traversed.begin(), files_traversed.end(), find_if_traversed);
        if(it == files_traversed.end())
        {
            look_for_includes_in_file(file_to_traverse, files_to_traverse);
            files_traversed.push_back(file_to_traverse);
        }
        files_to_traverse.pop();
    }

    return files_traversed;
}

void ShaderParser::look_for_includes_in_file(const std::filesystem::path& file, std::queue<std::filesystem::path>& files_to_traverse)
{
    const auto fileContent = utils::load_text_from_file(file.string());
    const auto absoluteDirectoryPath = file.parent_path();

    const auto words_begin = std::sregex_iterator(fileContent.begin(), fileContent.end(), regex);
    const auto words_end = std::sregex_iterator();
    for(std::sregex_iterator i = words_begin; i != words_end; ++i)
    {
        const std::string match_str = i->str();
        bool isPathAbsolute{false};
        const auto offset = find_path_beginning_sign_idx(match_str, isPathAbsolute);

        const auto first = offset + 1;
        const auto count = match_str.length() - (first + 1);
        const auto regexPath(match_str.substr(first, count));
        if(isPathAbsolute)
        {
            files_to_traverse.push(regexPath);
        }
        else
        {
            files_to_traverse.push((absoluteDirectoryPath / regexPath).lexically_normal());
        }
    }
}

std::string::size_type ShaderParser::find_path_beginning_sign_idx(const std::string& includeLine, bool& isPathAbsolute)
{
    for(std::string::size_type i = 0; i < includeLine.size(); i++)
    {
        if(includeLine[i] == '"')
        {
            isPathAbsolute = false;
            return i;
        }
        if(includeLine[i] == '<')
        {
            isPathAbsolute = true;
            return i;
        }
    }

    return std::string::npos;
};

std::map<GLenum, std::string> ShaderParser::preProcess(const std::string& shaderSource)
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

        const size_t shaderBodyStartPos = shaderSource.find_first_not_of("\r\n", eol);
        const size_t shaderBodyEndPos = shaderSource.find(typeToken, shaderBodyStartPos);
        const size_t shaderBodySize = shaderBodyEndPos - (shaderBodyStartPos == std::string::npos ? shaderSource.size() - 1 : shaderBodyStartPos);
        auto shaderBody = shaderSource.substr(shaderBodyStartPos, shaderBodySize);
        shaderSources.emplace(shaderTypeFromString(type), std::move(shaderBody));

        pos = shaderBodyEndPos;
    }

    return shaderSources;
}

GLenum ShaderParser::shaderTypeFromString(const std::string& type)
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
    if(type == "compute")
    {
        return GL_COMPUTE_SHADER;
    }

    SPARK_ERROR("Unknown shader type! {}", type);
    return 0;
}
}  // namespace spark::utils