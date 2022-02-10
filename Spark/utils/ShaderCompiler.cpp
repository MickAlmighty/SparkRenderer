#include "ShaderCompiler.hpp"

#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <shaderc/shaderc.hpp>

#include "Logging.h"

namespace
{
class GlslShaderIncluder : public shaderc::CompileOptions::IncluderInterface
{
    private:
    struct IncludeMetadata
    {
        std::string aboslutePath;
        std::string fileContent;
    };

    public:
    // Handles shaderc_include_resolver_fn callbacks.
    virtual shaderc_include_result* GetInclude(const char* requested_source, shaderc_include_type include_type, const char* requesting_source,
                                               size_t include_depth) override
    {
        auto* includeMetadata = new IncludeMetadata();
        if(include_type == shaderc_include_type_relative)  // E.g. #include "source"
        {
            const auto dirAbs = std::filesystem::path(requesting_source).parent_path();
            includeMetadata->aboslutePath = (dirAbs / requested_source).lexically_normal().string();
        }
        else if(include_type == shaderc_include_type_standard)  // E.g. #include <source>)
        {
            includeMetadata->aboslutePath = requested_source;
        }

        auto* data = new shaderc_include_result();
        try
        {
            std::stringstream fileStream;
            std::ifstream shaderFile;
            shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            shaderFile.open(includeMetadata->aboslutePath);
            fileStream << shaderFile.rdbuf();
            shaderFile.close();

            includeMetadata->fileContent = fileStream.str();
        }
        catch(const std::ifstream::failure& e)
        {
            // error message
            includeMetadata->fileContent = "FILE_NOT_SUCCESSFULLY_READ";
            includeMetadata->aboslutePath.clear();

            fill_shaderc_include_result(data, includeMetadata);
            return data;
        }

        fill_shaderc_include_result(data, includeMetadata);
        return data;
    };

    // Handles shaderc_include_result_release_fn callbacks.
    void ReleaseInclude(shaderc_include_result* data) override
    {
        auto* includeMetadata = reinterpret_cast<IncludeMetadata*>(data->user_data);
        delete includeMetadata;
        delete data;
    };

    void fill_shaderc_include_result(shaderc_include_result* data, IncludeMetadata* metadata)
    {
        data->user_data = metadata;
        data->content = metadata->fileContent.c_str();
        data->content_length = metadata->fileContent.length();
        data->source_name = metadata->aboslutePath.c_str();
        data->source_name_length = metadata->aboslutePath.length();
    }
};

shaderc_shader_kind select_proper_shader_kind(GLenum glShaderType)
{
    switch(glShaderType)
    {
        case GL_VERTEX_SHADER:
            return shaderc_vertex_shader;
        case GL_TESS_CONTROL_SHADER:
            return shaderc_tess_control_shader;
        case GL_TESS_EVALUATION_SHADER:
            return shaderc_tess_evaluation_shader;
        case GL_GEOMETRY_SHADER:
            return shaderc_geometry_shader;
        case GL_FRAGMENT_SHADER:
            return shaderc_fragment_shader;
        case GL_COMPUTE_SHADER:
            return shaderc_compute_shader;
    }

    return shaderc_glsl_infer_from_source;
}
}  // namespace

namespace spark::utils
{
std::vector<unsigned> ShaderCompiler::compile(const std::string& source_name, GLenum glShaderType, const std::string& source, bool optimize)
{
    const shaderc::Compiler compiler{};
    shaderc::CompileOptions options;

    if(optimize)
        options.SetOptimizationLevel(shaderc_optimization_level_performance);

    //options.SetGenerateDebugInfo();
    options.SetTargetEnvironment(shaderc_target_env_opengl, shaderc_env_version_opengl_4_5);
    options.SetIncluder(std::make_unique<GlslShaderIncluder>());

    const shaderc::SpvCompilationResult module =
        compiler.CompileGlslToSpv(source, select_proper_shader_kind(glShaderType), source_name.c_str(), options);

    if(module.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        SPARK_ERROR(module.GetErrorMessage());

        throw std::runtime_error("Shader compilation failed!");
    }

    return {module.cbegin(), module.cend()};
}
}  // namespace spark::utils