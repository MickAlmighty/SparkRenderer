#pragma once

#include <filesystem>
#include <memory>

namespace spark::resourceManagement
{
class Resource;
}

namespace spark::loaders
{
class ShaderLoader final
{
    public:
    static std::shared_ptr<resourceManagement::Resource> load(const std::filesystem::path& resourcesRootPath,
                                                                      const std::filesystem::path& resourceRelativePath);

    static bool isExtensionSupported(const std::string& ext);
    static std::vector<std::string> supportedExtensions();

    ShaderLoader(const ShaderLoader&) = delete;
    ShaderLoader(const ShaderLoader&&) = delete;
    ShaderLoader& operator=(const ShaderLoader&) = delete;
    ShaderLoader& operator=(const ShaderLoader&&) = delete;

    private:
    ShaderLoader() = default;
    ~ShaderLoader() = default;
};
}  // namespace spark::loaders