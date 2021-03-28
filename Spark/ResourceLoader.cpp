#include "ResourceLoader.h"

#include <stb_image/stb_image.h>

#include "CommonUtils.h"
#include "Structs.h"
#include "Logging.h"

namespace spark
{
using Path = std::filesystem::path;

std::optional<std::shared_ptr<PbrCubemapTexture>> ResourceLoader::loadHdrTexture(const std::string& path)
{
    stbi_set_flip_vertically_on_load(true);
    int width, height, nrComponents;
    float* data = stbi_loadf(path.c_str(), &width, &height, &nrComponents, 0);

    if(!data)
    {
        SPARK_ERROR("Failed to load HDR image '{}'.", path);
        return std::nullopt;
    }

    GLuint hdrTexture{0};
    utils::createTexture2D(hdrTexture, width, height, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, false, data);

    stbi_image_free(data);

    auto tex = std::make_shared<PbrCubemapTexture>(hdrTexture, path, 1024);
    glDeleteTextures(1, &hdrTexture);
    return {tex};
}
}  // namespace spark
