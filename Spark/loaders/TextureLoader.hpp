#pragma once

#include <filesystem>
#include <memory>
#include <vector>

namespace spark::resourceManagement
{
class Resource;
}

namespace spark::loaders
{
class TextureLoader final
{
    public:
    static std::shared_ptr<resourceManagement::Resource> load(const std::filesystem::path& resourcesRootPath,
                                                              const std::filesystem::path& resourceRelativePath);

    static bool isExtensionSupported(const std::string& ext);
    static std::vector<std::string> supportedExtensions();

    TextureLoader(const TextureLoader&) = delete;
    TextureLoader(const TextureLoader&&) = delete;
    TextureLoader& operator=(const TextureLoader&) = delete;
    TextureLoader& operator=(const TextureLoader&&) = delete;

    private:
    TextureLoader() = default;
    ~TextureLoader() = default;

    static bool isTextureUnCompressed(const std::string& extension);
    static bool isTextureCompressed(const std::string& extension);

    static std::shared_ptr<resourceManagement::Resource> loadCompressed(const std::filesystem::path& resourcesRootPath,
                                                                        const std::filesystem::path& resourceRelativePath);

    static std::shared_ptr<resourceManagement::Resource> loadUnCompressed(const std::filesystem::path& resourcesRootPath,
                                                                          const std::filesystem::path& resourceRelativePath);

    static std::shared_ptr<resourceManagement::Resource> loadHdrTexture(const std::filesystem::path& resourcesRootPath,
                                                                          const std::filesystem::path& resourceRelativePath);


    static const std::vector<std::string> extsUnCompressed;
    static const std::vector<std::string> extsCompressed;
    static const std::string hdrExtension;

};
}  // namespace spark::loaders