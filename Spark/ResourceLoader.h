#pragma once

#include <filesystem>
#include <optional>

namespace spark
{
struct PbrCubemapTexture;

class ResourceLoader final
{
    public:
    static std::optional<std::shared_ptr<PbrCubemapTexture>> loadHdrTexture(const std::string& path);

    ResourceLoader(const ResourceLoader&) = delete;
    ResourceLoader(const ResourceLoader&&) = delete;
    ResourceLoader& operator=(const ResourceLoader&) = delete;
    ResourceLoader& operator=(const ResourceLoader&&) = delete;

    private:
    ResourceLoader() = default;
    ~ResourceLoader() = default;
};

}  // namespace spark