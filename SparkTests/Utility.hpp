#pragma once
#include <filesystem>

namespace utility
{
inline std::filesystem::path findFileOrDirectory(const std::string& name)
{
    auto currentPath = std::filesystem::current_path();

    for (int i = 0; i < 5; ++i)
    {
        if (std::filesystem::exists(currentPath / name))
        {
            return (currentPath / name);
        }

        if (currentPath.has_parent_path())
        {
            currentPath = currentPath.parent_path();
            continue;
        }

        throw std::runtime_error("Path has not been found!");
    }

    throw std::runtime_error("Path has not been found!");
}
} // namespace utility

