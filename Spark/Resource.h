#pragma once
#include <filesystem>

namespace spark::resourceManagement
{
class Resource abstract
{
    public:
    Resource(const std::filesystem::path& path_) : path(path_) {}
    virtual ~Resource() = default;

    std::filesystem::path getPath() const
    {
        return path;
    }

    protected:
    std::filesystem::path path{};
};
}  // namespace spark::resourceManagement