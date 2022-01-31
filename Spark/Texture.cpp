#include "Texture.h"

namespace spark::resources
{
Texture::Texture(const std::filesystem::path& path_, utils::UniqueTextureHandle textureHandle_, int width_, int height_)
    : Resource(path_), textureHandle(std::move(textureHandle_)), width(width_), height(height_)
{
}

GLuint Texture::getID() const
{
    return textureHandle.get();
}

}  // namespace spark::resources
