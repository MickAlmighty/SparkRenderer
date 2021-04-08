#include "Texture.h"

namespace spark::resources
{
Texture::Texture(const std::filesystem::path& path_, GLuint id_, int width_, int height_) : Resource(path_), ID(id_), width(width_), height(height_)
{
}

Texture::~Texture()
{
    glDeleteTextures(1, &ID);
}

GLuint Texture::getID() const
{
    return ID;
}

}  // namespace spark::resources
