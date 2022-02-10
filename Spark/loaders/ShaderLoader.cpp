#include "ShaderLoader.hpp"

#include <fstream>
#include <istream>
#include <sstream>
#include <shaderc/shaderc.hpp>

#include "Logging.h"
#include "Shader.h"
#include "utils/ShaderCompiler.hpp"
#include "utils/ShaderParser.hpp"

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

std::vector<unsigned> readShaderBinaryFromFile(const std::filesystem::path& path)
{
    std::vector<unsigned> binary;

    if(std::ifstream input(path, std::ios::in | std::ios::binary); input.is_open())
    {
        // get its size:
        input.seekg(0, std::ios::end);
        const auto fileSize = input.tellg();
        input.seekg(0, std::ios::beg);

        if(fileSize != 0)
        {
            // read the data:
            binary.resize(fileSize / 4);
            input.read(reinterpret_cast<char*>(&binary[0]), fileSize);
        }

        input.close();
    }

    return binary;
}

void write_binary_to_file(const std::filesystem::path& binary_path, const std::vector<unsigned>& binary)
{
    if(std::ofstream output(binary_path, std::ios::out | std::ios::binary | std::ios::trunc); output.is_open())
    {
        output.write(reinterpret_cast<const char*>(binary.data()), binary.size() * sizeof(unsigned));
        output.close();
    }
}

bool is_compilation_needed(const std::filesystem::path& shaderPath, const std::filesystem::path& binary_path)
{
    return !std::filesystem::exists(binary_path) || spark::utils::ShaderParser::any_of_files_from_inlude_chain_modified(shaderPath, binary_path);
}
}  // namespace

namespace spark::loaders
{
std::shared_ptr<resourceManagement::Resource> ShaderLoader::load(const std::filesystem::path& resourcesRootPath,
                                                                 const std::filesystem::path& resourceRelativePath)
{
    const auto path = resourcesRootPath / resourceRelativePath;
    SPARK_DEBUG("Loading Shader: {}", resourceRelativePath.string());
    const auto shaderHashName = getHashedFileNameFromPath(resourceRelativePath);

    const auto shaderSources = utils::ShaderParser::parseShaderFile(path.string());
    std::vector<std::pair<GLenum, std::vector<unsigned>>> shaders{};
    shaders.reserve(shaderSources.size());
    for(const auto& [glShaderType, source] : shaderSources)
    {
        const auto cachedShaderName = shaderHashName + select_proper_shader_suffix(glShaderType);
        const auto cachedShaderAbsolutePath = resourcesRootPath / "cache" / cachedShaderName;

        if(is_compilation_needed(path, cachedShaderAbsolutePath))
        {
            const auto binary = utils::ShaderCompiler::compile(path.string().c_str(), glShaderType, source, false);
            write_binary_to_file(cachedShaderAbsolutePath, binary);

            shaders.emplace_back(glShaderType, binary);
        }
        else
        {
            shaders.emplace_back(glShaderType, readShaderBinaryFromFile(cachedShaderAbsolutePath));
        }
    }

    return std::make_shared<resources::Shader>(path, shaders);
    //return std::make_shared<resources::Shader>(path);
}

bool ShaderLoader::isExtensionSupported(const std::string& ext)
{
    const auto supportedExts = supportedExtensions();
    const auto it = std::find_if(supportedExts.begin(), supportedExts.end(), [&ext](const auto& e) { return e == ext; });
    return it != supportedExts.end();
}

std::vector<std::string> ShaderLoader::supportedExtensions()
{
    return {".glsl"};
}
}  // namespace spark::loaders