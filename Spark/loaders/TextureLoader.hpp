#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "ILoader.hpp"

namespace spark::loaders
{
class TextureLoader final : public ILoader
{
    public:
    std::shared_ptr<resourceManagement::Resource> load(const std::filesystem::path& resourcesRootPath,
                                                       const std::filesystem::path& resourceRelativePath) const override;

    bool isExtensionSupported(const std::string& ext) const override;
    std::vector<std::string> supportedExtensions() const override;

    TextureLoader() = default;
    ~TextureLoader() = default;
    TextureLoader(const TextureLoader&) = delete;
    TextureLoader(const TextureLoader&&) = delete;
    TextureLoader& operator=(const TextureLoader&) = delete;
    TextureLoader& operator=(const TextureLoader&&) = delete;

    private:
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