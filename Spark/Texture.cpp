#include "Texture.h"

#include <gli/gli.hpp>
#include <stb_image/stb_image.h>

#include "Logging.h"

namespace spark::resources
{
Texture::Texture(const resourceManagement::ResourceIdentifier& identifier)
    : Resource(identifier)
{
    const std::string ext = id.getResourceExtension().string();
    if (ext == ".png" || ext == ".jpg" || ext == "tga")
    {
        isTextureCompressed = false;
    }
    if (ext == ".dds" || ext == ".DDS" || ext == ".ktx" || ext == ".KTX")
    {
        isTextureCompressed = true;
    }
}

bool Texture::isResourceReady()
{
    return isLoadedIntoRAM() && isLoadedIntoDeviceMemory();
}

bool Texture::gpuLoad()
{
    //Timer timer("Allocating gpu memory for texture: " + id.getFullPath().string());
    const std::string ext = id.getResourceExtension().string();
    if (isTextureCompressed)
    {
        createGpuCompressedTexture();
    }
    else
    {
        createGpuTexture();
    }

    setLoadedIntoDeviceMemory(true);
    return true;
}

bool Texture::gpuUnload()
{
    //Timer timer("Freeing gpu memory for texture: " + id.getFullPath().string());
    glDeleteTextures(1, &ID);

    setLoadedIntoDeviceMemory(false);
    return true;
}

bool Texture::load()
{
    //Timer timer("Loading texture from file: " + id.getFullPath().string());
    const std::string ext = id.getResourceExtension().string();
    if (isTextureCompressed)
    {
        compressedTextureData = loadCompressedTextureData(id.getFullPath());
    }
    else
    {
        texturePixelArray = loadTextureData(id.getFullPath());
    }

    setLoadedIntoRam(true);
    return true;
}

bool Texture::unload()
{
    //Timer timer("Unloading texture:" + id.getFullPath().string());
    if (texturePixelArray)
    {
        stbi_image_free(texturePixelArray);
    }
    if (!compressedTextureData.empty())
    {
        compressedTextureData.clear();
    }

    setLoadedIntoRam(false);
    return true;
}

GLuint Texture::getID() const
{
    return ID;
}

unsigned char* Texture::loadTextureData(const std::filesystem::path& filePath)
{
    unsigned char* pixels = nullptr;
    pixels = stbi_load(filePath.string().c_str(), &width, &height, &channels, 0);

    if (pixels == nullptr)
    {
        SPARK_ERROR("Texture from path: " + filePath.string() + " cannot be loaded!");
        return nullptr;
    }

    return pixels;
}

gli::texture Texture::loadCompressedTextureData(const std::filesystem::path& filePath) const
{
    return gli::load(filePath.string());
}

void Texture::createGpuTexture()
{
    GLenum format{};
    switch (channels)
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
        return;
    ;
    }

    GLuint texture;
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, texturePixelArray);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    ID = texture;
}

void Texture::createGpuCompressedTexture()
{
    if (compressedTextureData.empty())
        return;

    gli::gl GL(gli::gl::PROFILE_GL33);
    gli::gl::format const Format = GL.translate(compressedTextureData.format(), compressedTextureData.swizzles());
    GLenum Target = GL.translate(compressedTextureData.target());

    GLuint textureID = 0;
    glGenTextures(1, &textureID);
    glBindTexture(Target, textureID);
    glTexParameteri(Target, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(Target, GL_TEXTURE_MAX_LEVEL, static_cast<GLint>(compressedTextureData.levels() - 1));
    glTexParameteri(Target, GL_TEXTURE_SWIZZLE_R, Format.Swizzles[0]);
    glTexParameteri(Target, GL_TEXTURE_SWIZZLE_G, Format.Swizzles[1]);
    glTexParameteri(Target, GL_TEXTURE_SWIZZLE_B, Format.Swizzles[2]);
    glTexParameteri(Target, GL_TEXTURE_SWIZZLE_A, Format.Swizzles[3]);
    float maxAnisotropicFiltering{ 1.0f };
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAnisotropicFiltering);
    glTexParameterf(Target, GL_TEXTURE_MAX_ANISOTROPY, maxAnisotropicFiltering);

    glm::tvec3<GLsizei> const Extent(compressedTextureData.extent());
    GLsizei const FaceTotal = static_cast<GLsizei>(compressedTextureData.layers() * compressedTextureData.faces());

    switch (compressedTextureData.target())
    {
    case gli::TARGET_1D:
        glTexStorage1D(Target, static_cast<GLint>(compressedTextureData.levels()), Format.Internal, Extent.x);
        break;
    case gli::TARGET_1D_ARRAY:
    case gli::TARGET_2D:
    case gli::TARGET_CUBE:
        glTexStorage2D(Target, static_cast<GLint>(compressedTextureData.levels()), Format.Internal, Extent.x,
            compressedTextureData.target() == gli::TARGET_2D ? Extent.y : FaceTotal);
        break;
    case gli::TARGET_2D_ARRAY:
    case gli::TARGET_3D:
    case gli::TARGET_CUBE_ARRAY:
        glTexStorage3D(Target, static_cast<GLint>(compressedTextureData.levels()), Format.Internal, Extent.x, Extent.y,
            compressedTextureData.target() == gli::TARGET_3D ? Extent.z : FaceTotal);
        break;
    default:
        assert(0);
        break;
    }

    for (std::size_t Layer = 0; Layer < compressedTextureData.layers(); ++Layer)
    {
        for (std::size_t Face = 0; Face < compressedTextureData.faces(); ++Face)
        {
            for (std::size_t Level = 0; Level < compressedTextureData.levels(); ++Level)
            {
                GLsizei const LayerGL = static_cast<GLsizei>(Layer);
                glm::tvec3<GLsizei> Extent(compressedTextureData.extent(Level));
                Target = gli::is_target_cube(compressedTextureData.target()) ? static_cast<GLenum>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + Face) : Target;

                switch (compressedTextureData.target())
                {
                case gli::TARGET_1D:
                    if (gli::is_compressed(compressedTextureData.format()))
                        glCompressedTexSubImage1D(Target, static_cast<GLint>(Level), 0, Extent.x, Format.Internal,
                            static_cast<GLsizei>(compressedTextureData.size(Level)), compressedTextureData.data(Layer, Face, Level));
                    else
                        glTexSubImage1D(Target, static_cast<GLint>(Level), 0, Extent.x, Format.External, Format.Type,
                            compressedTextureData.data(Layer, Face, Level));
                    break;
                case gli::TARGET_1D_ARRAY:
                case gli::TARGET_2D:
                case gli::TARGET_CUBE:
                    if (gli::is_compressed(compressedTextureData.format()))
                        glCompressedTexSubImage2D(Target, static_cast<GLint>(Level), 0, 0, Extent.x,
                            compressedTextureData.target() == gli::TARGET_1D_ARRAY ? LayerGL : Extent.y, Format.Internal,
                            static_cast<GLsizei>(compressedTextureData.size(Level)), compressedTextureData.data(Layer, Face, Level));
                    else
                        glTexSubImage2D(Target, static_cast<GLint>(Level), 0, 0, Extent.x,
                            compressedTextureData.target() == gli::TARGET_1D_ARRAY ? LayerGL : Extent.y, Format.External, Format.Type,
                            compressedTextureData.data(Layer, Face, Level));
                    break;
                case gli::TARGET_2D_ARRAY:
                case gli::TARGET_3D:
                case gli::TARGET_CUBE_ARRAY:
                    if (gli::is_compressed(compressedTextureData.format()))
                        glCompressedTexSubImage3D(Target, static_cast<GLint>(Level), 0, 0, 0, Extent.x, Extent.y,
                            compressedTextureData.target() == gli::TARGET_3D ? Extent.z : LayerGL, Format.Internal,
                            static_cast<GLsizei>(compressedTextureData.size(Level)), compressedTextureData.data(Layer, Face, Level));
                    else
                        glTexSubImage3D(Target, static_cast<GLint>(Level), 0, 0, 0, Extent.x, Extent.y,
                            compressedTextureData.target() == gli::TARGET_3D ? Extent.z : LayerGL, Format.External, Format.Type,
                            compressedTextureData.data(Layer, Face, Level));
                    break;
                default:
                    assert(0);
                    break;
                }
            }
        }
    }

    ID = textureID;
}
}
