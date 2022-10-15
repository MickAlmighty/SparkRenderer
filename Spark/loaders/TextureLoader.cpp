#include "TextureLoader.hpp"

#include <array>
#include <algorithm>

#include <gli/load.hpp>
#include <stb_image.h>

#include "utils/CommonUtils.h"
#include "glad_glfw3.h"
#include "Logging.h"
#include "stb_image.h"

#include "Texture.h"

namespace spark::loaders
{
const std::vector<std::string> TextureLoader::extsUnCompressed{".png", ".jpg", ".tga"};
const std::vector<std::string> TextureLoader::extsCompressed{".dds", ".ktx"};
const std::string TextureLoader::hdrExtension{".hdr"};

std::shared_ptr<resourceManagement::Resource> TextureLoader::load(const std::filesystem::path& resourcesRootPath,
                                                                  const std::filesystem::path& resourceRelativePath) const
{
    const auto ext = utils::toLowerCase(resourceRelativePath.extension().string());
    if(isTextureUnCompressed(ext))
    {
        return loadUnCompressed(resourcesRootPath, resourceRelativePath);
    }

    if(isTextureCompressed(ext))
    {
        return loadCompressed(resourcesRootPath, resourceRelativePath);
    }

    if(ext == ".hdr")
    {
        return loadHdrTexture(resourcesRootPath, resourceRelativePath);
    }

    SPARK_ERROR("Unsupported extension! {}", ext);
    return nullptr;
}

bool TextureLoader::isExtensionSupported(const std::string& ext) const
{
    const auto supportedExts = supportedExtensions();
    const auto it = std::find_if(supportedExts.begin(), supportedExts.end(), [&ext](const auto& e) { return e == ext; });
    return it != supportedExts.end();
}

std::vector<std::string> TextureLoader::supportedExtensions() const
{
    std::vector<std::string> supportedExtensions;
    supportedExtensions.reserve(extsUnCompressed.size() + extsCompressed.size() + 1);

    std::copy(extsUnCompressed.begin(), extsUnCompressed.end(), std::back_inserter(supportedExtensions));
    std::copy(extsCompressed.begin(), extsCompressed.end(), std::back_inserter(supportedExtensions));
    supportedExtensions.push_back(hdrExtension);

    return supportedExtensions;
}

bool TextureLoader::isTextureUnCompressed(const std::string& extension)
{
    const auto it = std::find_if(extsUnCompressed.begin(), extsUnCompressed.end(), [&extension](const auto& ext) { return ext == extension; });
    return it != extsUnCompressed.end();
}

bool TextureLoader::isTextureCompressed(const std::string& extension)
{
    const auto it = std::find_if(extsCompressed.begin(), extsCompressed.end(), [&extension](const auto& ext) { return ext == extension; });
    return it != extsCompressed.end();
}

std::shared_ptr<resourceManagement::Resource> TextureLoader::loadCompressed(const std::filesystem::path& resourcesRootPath,
                                                                            const std::filesystem::path& resourceRelativePath)
{
    const auto path = (resourcesRootPath / resourceRelativePath).string();
    const auto gliTexture = gli::load(path);

    gli::gl GL(gli::gl::PROFILE_GL33);
    gli::gl::format const Format = GL.translate(gliTexture.format(), gliTexture.swizzles());
    GLenum Target = GL.translate(gliTexture.target());

    GLuint textureID = 0;
    glGenTextures(1, &textureID);
    glBindTexture(Target, textureID);
    glTexParameteri(Target, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(Target, GL_TEXTURE_MAX_LEVEL, static_cast<GLint>(gliTexture.levels() - 1));
    glTexParameteri(Target, GL_TEXTURE_SWIZZLE_R, Format.Swizzles[0]);
    glTexParameteri(Target, GL_TEXTURE_SWIZZLE_G, Format.Swizzles[1]);
    glTexParameteri(Target, GL_TEXTURE_SWIZZLE_B, Format.Swizzles[2]);
    glTexParameteri(Target, GL_TEXTURE_SWIZZLE_A, Format.Swizzles[3]);
    float maxAnisotropicFiltering{1.0f};
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAnisotropicFiltering);
    glTexParameterf(Target, GL_TEXTURE_MAX_ANISOTROPY, maxAnisotropicFiltering);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    glm::tvec3<GLsizei> const Extent(gliTexture.extent());
    GLsizei const FaceTotal = static_cast<GLsizei>(gliTexture.layers() * gliTexture.faces());

    switch(gliTexture.target())
    {
        case gli::TARGET_1D:
            glTexStorage1D(Target, static_cast<GLint>(gliTexture.levels()), Format.Internal, Extent.x);
            break;
        case gli::TARGET_1D_ARRAY:
        case gli::TARGET_2D:
        case gli::TARGET_CUBE:
            glTexStorage2D(Target, static_cast<GLint>(gliTexture.levels()), Format.Internal, Extent.x,
                           gliTexture.target() == gli::TARGET_2D ? Extent.y : FaceTotal);
            break;
        case gli::TARGET_2D_ARRAY:
        case gli::TARGET_3D:
        case gli::TARGET_CUBE_ARRAY:
            glTexStorage3D(Target, static_cast<GLint>(gliTexture.levels()), Format.Internal, Extent.x, Extent.y,
                           gliTexture.target() == gli::TARGET_3D ? Extent.z : FaceTotal);
            break;
        default:
            assert(0);
            break;
    }

    for(std::size_t Layer = 0; Layer < gliTexture.layers(); ++Layer)
    {
        for(std::size_t Face = 0; Face < gliTexture.faces(); ++Face)
        {
            for(std::size_t Level = 0; Level < gliTexture.levels(); ++Level)
            {
                GLsizei const LayerGL = static_cast<GLsizei>(Layer);
                glm::tvec3<GLsizei> Extent(gliTexture.extent(Level));
                Target = gli::is_target_cube(gliTexture.target()) ? static_cast<GLenum>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + Face) : Target;

                switch(gliTexture.target())
                {
                    case gli::TARGET_1D:
                        if(gli::is_compressed(gliTexture.format()))
                            glCompressedTexSubImage1D(Target, static_cast<GLint>(Level), 0, Extent.x, Format.Internal,
                                                      static_cast<GLsizei>(gliTexture.size(Level)), gliTexture.data(Layer, Face, Level));
                        else
                            glTexSubImage1D(Target, static_cast<GLint>(Level), 0, Extent.x, Format.External, Format.Type,
                                            gliTexture.data(Layer, Face, Level));
                        break;
                    case gli::TARGET_1D_ARRAY:
                    case gli::TARGET_2D:
                    case gli::TARGET_CUBE:
                        if(gli::is_compressed(gliTexture.format()))
                            glCompressedTexSubImage2D(Target, static_cast<GLint>(Level), 0, 0, Extent.x,
                                                      gliTexture.target() == gli::TARGET_1D_ARRAY ? LayerGL : Extent.y, Format.Internal,
                                                      static_cast<GLsizei>(gliTexture.size(Level)), gliTexture.data(Layer, Face, Level));
                        else
                            glTexSubImage2D(Target, static_cast<GLint>(Level), 0, 0, Extent.x,
                                            gliTexture.target() == gli::TARGET_1D_ARRAY ? LayerGL : Extent.y, Format.External, Format.Type,
                                            gliTexture.data(Layer, Face, Level));
                        break;
                    case gli::TARGET_2D_ARRAY:
                    case gli::TARGET_3D:
                    case gli::TARGET_CUBE_ARRAY:
                        if(gli::is_compressed(gliTexture.format()))
                            glCompressedTexSubImage3D(Target, static_cast<GLint>(Level), 0, 0, 0, Extent.x, Extent.y,
                                                      gliTexture.target() == gli::TARGET_3D ? Extent.z : LayerGL, Format.Internal,
                                                      static_cast<GLsizei>(gliTexture.size(Level)), gliTexture.data(Layer, Face, Level));
                        else
                            glTexSubImage3D(Target, static_cast<GLint>(Level), 0, 0, 0, Extent.x, Extent.y,
                                            gliTexture.target() == gli::TARGET_3D ? Extent.z : LayerGL, Format.External, Format.Type,
                                            gliTexture.data(Layer, Face, Level));
                        break;
                    default:
                        assert(0);
                        break;
                }
            }
        }
    }

    return std::make_shared<resources::Texture>(resourceRelativePath, utils::UniqueTextureHandle(textureID), Extent.x, Extent.y);
}

std::shared_ptr<resourceManagement::Resource> TextureLoader::loadUnCompressed(const std::filesystem::path& resourcesRootPath,
                                                                              const std::filesystem::path& resourceRelativePath)
{
    const auto path = (resourcesRootPath / resourceRelativePath).string();
    int width{0}, height{0}, channels{0};
    unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels, 0);

    if(pixels == nullptr)
    {
        SPARK_ERROR("Texture from path: " + path + " cannot be loaded!");
        return nullptr;
    }

    GLenum format{};
    switch(channels)
    {
        case(1):
        {
            format = GL_RED;
            break;
        }
        case(2):
        {
            format = GL_RG;
            break;
        }
        case(3):
        {
            format = GL_RGB;
            break;
        }
        case(4):
        {
            format = GL_RGBA;
            break;
        }
        default:
            SPARK_ERROR("Invalid number of channels to create Opengl Texture!");
            return nullptr;
    }

    auto texId = utils::createTexture2D(width, height, format, format, GL_UNSIGNED_BYTE, GL_REPEAT, GL_LINEAR, true, pixels);
    stbi_image_free(pixels);

    return std::make_shared<resources::Texture>(resourceRelativePath, std::move(texId), width, height);
}

std::shared_ptr<resourceManagement::Resource> TextureLoader::loadHdrTexture(const std::filesystem::path& resourcesRootPath,
                                                                            const std::filesystem::path& resourceRelativePath)
{
    const auto path = (resourcesRootPath / resourceRelativePath).string();
    int width, height, nrComponents;
    float* data = stbi_loadf(path.c_str(), &width, &height, &nrComponents, 0);

    if(!data)
    {
        SPARK_ERROR("Failed to load HDR image '{}'.", path);
        return nullptr;
    }

    auto hdrTexture = utils::createTexture2D(width, height, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, false, data);

    stbi_image_free(data);

    return std::make_shared<resources::Texture>(resourceRelativePath, std::move(hdrTexture), width, height);
}
}  // namespace spark::loaders