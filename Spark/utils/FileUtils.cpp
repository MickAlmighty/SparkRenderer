#include "FileUtils.hpp"

#include <fstream>
#include <istream>
#include <sstream>

#include "Logging.h"

namespace spark::utils
{
void write_text_to_file(const std::filesystem::path& path, const std::string& text)
{
    if(std::ofstream output(path, std::ios::out | std::ios::trunc); output.is_open())
    {
        output.write(text.c_str(), text.size());
        output.close();
    }
}

std::string load_text_from_file(const std::filesystem::path& path)
{
    std::ifstream shaderFile;
    try
    {
        shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        shaderFile.open(path);
        std::stringstream shaderStream;
        shaderStream << shaderFile.rdbuf();
        shaderFile.close();

        return shaderStream.str();
    }
    catch(const std::ifstream::failure& e)
    {
        shaderFile.close();
        SPARK_ERROR("SHADER::FILE_NOT_SUCCESSFULLY_READ: {}, {}", e.what(), path.string());
        return {};
    }
}
}  // namespace spark::utils