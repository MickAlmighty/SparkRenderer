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
    std::shared_ptr<resourceManagement::Resource> createModel(
        const std::shared_ptr<resourceManagement::ResourceIdentifier>& resourceIdentifier) const;

    private:

};
}  // namespace spark
