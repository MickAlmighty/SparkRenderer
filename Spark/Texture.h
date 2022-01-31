#pragma once

#include <glad/glad.h>

#include "Resource.h"
#include "utils/GlHandle.hpp"

namespace spark::resources
{
class Texture : public resourceManagement::Resource
{
    public:
    Texture(const std::filesystem::path& path_, utils::UniqueTextureHandle textureHandle_, int width_, int height_);
    ~Texture() override = default;

    GLuint getID() const;

    private:
    utils::UniqueTextureHandle textureHandle{};
    int width{0}, height{0};
};
}  // namespace spark::resources