#pragma once
#include <filesystem>
#include <utility>

namespace spark::resourceManagement
{
class Resource
{
    public:
    Resource() = default;
    Resource(std::filesystem::path path_) : path(std::move(path_)) {}
    virtual ~Resource() = default;

    std::filesystem::path getPath() const
    {
        return path;
    }

    void setPath(const std::filesystem::path& p)
    {
        path = p;
    }

    protected:
    std::filesystem::path path{};
};
}  // namespace spark::resourceManagement