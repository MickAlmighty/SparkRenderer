#include "ShaderLoader.hpp"

#include <sstream>

#include <shaderc/shaderc.hpp>
#include <spirv_glsl.hpp>

#include "Logging.h"
#include "Shader.h"
#include "utils/ShaderCompiler.hpp"
#include "utils/ShaderParser.hpp"
#include "utils/FileUtils.hpp"

namespace
{
std::string select_proper_shader_suffix(GLenum glShaderType)
{
    switch(glShaderType)
    {
        case GL_VERTEX_SHADER:
            return "_vert";
        case GL_GEOMETRY_SHADER:
            return "_geom";
        case GL_TESS_CONTROL_SHADER:
            return "tess_control";
        case GL_TESS_EVALUATION_SHADER:
            return "tess_evaluation";
        case GL_FRAGMENT_SHADER:
            return "_frag";
        case GL_COMPUTE_SHADER:
            return "_comp";
    }

    return {};
}

std::string getHashedFileNameFromPath(const std::filesystem::path& path)
{
    std::stringstream s{};
    s << std::filesystem::hash_value(path);

    return s.str();
}

bool is_compilation_needed(const std::filesystem::path& shaderPath, const std::filesystem::path& binary_path)
{
    return !std::filesystem::exists(binary_path) || spark::utils::ShaderParser::any_of_files_from_inlude_chain_modified(shaderPath, binary_path);
}

std::string createCachedStagedShaderName(const std::string& shaderHashName, GLenum glShaderType)
{
    const auto shaderStageName = select_proper_shader_suffix(glShaderType);
    const auto cachedShaderName = shaderHashName + shaderStageName;
    const auto extension = ".glsl";
    return cachedShaderName + extension;
}
}  // namespace

namespace spark::loaders
{
std::shared_ptr<resourceManagement::Resource> ShaderLoader::load(const std::filesystem::path& resourcesRootPath,
                                                                 const std::filesystem::path& resourceRelativePath) const
{
    const auto path = resourcesRootPath / resourceRelativePath;
    const auto cacheAbsolutePath = resourcesRootPath / "cache";
    SPARK_DEBUG("Loading Shader: {}", resourceRelativePath.string());
    const auto shaderHashName = getHashedFileNameFromPath(resourceRelativePath);

    auto shaderSources = utils::ShaderParser::splitShaderSourceByStages(path.string());
    std::map<GLenum, std::string> shaders;
    for(auto& [glShaderType, source] : shaderSources)
    {
        const auto cachedShaderAbsolutePath = cacheAbsolutePath / createCachedStagedShaderName(shaderHashName, glShaderType);

        if(is_compilation_needed(path, cachedShaderAbsolutePath))
        {
            auto binary = utils::ShaderCompiler::compileVulkan(path.string().c_str(), glShaderType, source, false);
            spirv_cross::CompilerGLSL glslCompiler(std::move(binary));

            auto code = glslCompiler.compile();
            utils::write_text_to_file(cachedShaderAbsolutePath, code);
            shaders.emplace(glShaderType, std::move(code));
        }
        else
        {
            shaders.emplace(glShaderType, utils::load_text_from_file(cachedShaderAbsolutePath));
        }
    }

    return std::make_shared<resources::Shader>(path, shaders);
}

bool ShaderLoader::isExtensionSupported(const std::string& ext) const
{
    const auto supportedExts = supportedExtensions();
    const auto it = std::find_if(supportedExts.begin(), supportedExts.end(), [&ext](const auto& e) { return e == ext; });
    return it != supportedExts.end();
}

std::vector<std::string> ShaderLoader::supportedExtensions() const
{
    return {".glsl"};
}
}  // namespace spark::loaders