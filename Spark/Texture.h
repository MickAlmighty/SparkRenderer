#pragma once

#include <glad/glad.h>

#include "Resource.h"

namespace spark::resources
{
class Texture : public resourceManagement::Resource
{
    public:
    Texture(const std::filesystem::path& path_, GLuint id_, int width_, int height_);
    ~Texture() override;

    GLuint getID() const;

    private:
    GLuint ID{0};
    int width{0}, height{0};
};
}  // namespace spark::resources