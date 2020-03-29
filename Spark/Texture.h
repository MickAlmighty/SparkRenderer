#ifndef TEXTURE_H
#define TEXTURE_H
#include <glad/glad.h>
#include <gli/texture.hpp>


#include "GPUResource.h"
#include "Resource.h"

namespace spark::resources
{
class Texture : public resourceManagement::Resource, public resourceManagement::GPUResource
{
    public:
    Texture(const resourceManagement::ResourceIdentifier& identifier);

    bool isResourceReady() override;
    bool gpuLoad() override;
    bool gpuUnload() override;
    bool load() override;
    bool unload() override;

    GLuint getID() const;

    private:
    GLuint ID{0};
    bool compressedTexture{ false };
    int width{ 0 }, height{ 0 }, channels{ 0 };
    unsigned char* texturePixelArray{nullptr};
    gli::texture compressedTextureData{};

    [[nodiscard]] unsigned char* loadTextureData(const std::filesystem::path& filePath);
    [[nodiscard]] gli::texture loadCompressedTextureData(const std::filesystem::path& filePath) const;
    void createGpuTexture();
    void createGpuCompressedTexture();
};
}
#endif