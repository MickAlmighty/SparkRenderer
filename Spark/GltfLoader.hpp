#pragma once

#include <filesystem>
#include <map>

#include "Resource.h"

namespace spark
{
enum class TextureTarget : unsigned char;

namespace resourceManagement
{
    class ResourceIdentifier;
}

class Mesh;

class GltfLoader final
{
    public:
    std::shared_ptr<resourceManagement::Resource> createModel(const std::filesystem::path& resourcesRootPath,
                                                              const std::filesystem::path& resourceRelativePath) const;

    private:

};
}  // namespace spark
