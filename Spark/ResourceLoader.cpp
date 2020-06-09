#include "ResourceLoader.h"

#include <future>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <gli/gli.hpp>
#include <stb_image/stb_image.h>

#include "CommonUtils.h"
#include "Mesh.h"
#include "Shader.h"
#include "Spark.h"
#include "Structs.h"
#include "Timer.h"
#include "Logging.h"

namespace spark
{
using Path = std::filesystem::path;

std::map<std::string, std::vector<std::shared_ptr<Mesh>>> ResourceLoader::loadModels(std::filesystem::path& modelDirectory)
{
    std::map<std::string, std::vector<std::shared_ptr<Mesh>>> models;
    for(auto& path_it : std::filesystem::recursive_directory_iterator(modelDirectory))
    {
        if(checkExtension(path_it.path().extension().string(), ModelMeshExtensions))
        {
            models.emplace(path_it.path().string(), loadModel(path_it.path()));
        }
    }

    return models;
}

std::vector<std::shared_ptr<Mesh>> ResourceLoader::loadModel(const Path& path)
{
    Timer timer("ResourceLoader::loadModel( " + path.string() + " )");

    const auto scene = importScene(path);

    std::vector<std::shared_ptr<Mesh>> meshes = loadMeshes(scene, path);

    return meshes;
}

const aiScene* ResourceLoader::importScene(const std::filesystem::path& filePath)
{
    Assimp::Importer importer;
    const aiScene* scene = nullptr;

    scene = importer.ReadFile(filePath.string(), aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        SPARK_ERROR("{}", importer.GetErrorString());
        throw std::exception(importer.GetErrorString());
    }

    return scene;
}

std::vector<std::shared_ptr<Mesh>> ResourceLoader::loadMeshes(const aiScene* scene, const std::filesystem::path& modelPath)
{
    std::vector<std::shared_ptr<Mesh>> meshes;
    for(unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        meshes.push_back(loadMesh(scene->mMeshes[i], modelPath));
    }
    return meshes;
}

bool ResourceLoader::checkExtension(std::string&& extension, const std::vector<std::string>& extensions)
{
    const auto it = std::find(std::begin(extensions), std::end(extensions), extension);
    return it != std::end(extensions);
}

std::map<TextureTarget, Texture> ResourceLoader::findTextures(const std::filesystem::path& modelDirectory)
{
    std::map<TextureTarget, Texture> textures;
    for(auto& texture_path : std::filesystem::recursive_directory_iterator(modelDirectory))
    {
        size_t size = texture_path.path().string().find("_Diffuse");
        if(size != std::string::npos)
        {
            /*Texture tex = ResourceManager::getInstance()->findTexture(texture_path.path().string());
            textures.emplace(TextureTarget::DIFFUSE_TARGET, tex);*/
            continue;
        }

        size = texture_path.path().string().find("_Normal");
        if(size != std::string::npos)
        {
            /*Texture tex = ResourceManager::getInstance()->findTexture(texture_path.path().string());
            textures.emplace(TextureTarget::NORMAL_TARGET, tex);*/
        }

        size = texture_path.path().string().find("_Roughness");
        if(size != std::string::npos)
        {
            /*Texture tex = ResourceManager::getInstance()->findTexture(texture_path.path().string());
            textures.emplace(TextureTarget::ROUGHNESS_TARGET, tex);*/
        }

        size = texture_path.path().string().find("_Metalness");
        if(size != std::string::npos)
        {
            /*Texture tex = ResourceManager::getInstance()->findTexture(texture_path.path().string());
            textures.emplace(TextureTarget::METALNESS_TARGET, tex);*/
        }
    }

    return textures;
}

std::shared_ptr<Mesh> ResourceLoader::loadMesh(aiMesh* assimpMesh, const std::filesystem::path& modelPath)
{
    std::vector<glm::vec3> positions{assimpMesh->mNumVertices};
    std::vector<glm::vec3> normals{assimpMesh->mNumVertices};
    std::vector<glm::vec2> textureCoords{assimpMesh->mNumVertices};
    std::vector<glm::vec3> tangent{assimpMesh->mNumVertices};
    std::vector<glm::vec3> biTangent{assimpMesh->mNumVertices};


    for(unsigned int i = 0; i < assimpMesh->mNumVertices; i++)
    {
        positions[i].x = assimpMesh->mVertices[i].x;
        positions[i].y = assimpMesh->mVertices[i].y;
        positions[i].z = assimpMesh->mVertices[i].z;

        if(assimpMesh->HasNormals())
        {
            normals[i].x = assimpMesh->mNormals[i].x;
            normals[i].y = assimpMesh->mNormals[i].y;
            normals[i].z = assimpMesh->mNormals[i].z;
        }

        if(assimpMesh->HasTextureCoords(0))
        {
            textureCoords[i].x = assimpMesh->mTextureCoords[0][i].x;
            textureCoords[i].y = assimpMesh->mTextureCoords[0][i].y;
        }
        else
            textureCoords[i] = glm::vec2(0.0f, 0.0f);

        if(assimpMesh->HasTangentsAndBitangents())
        {
            tangent[i].x = assimpMesh->mTangents->x;
            tangent[i].y = assimpMesh->mTangents->y;
            tangent[i].z = assimpMesh->mTangents->z;

            biTangent[i].x = assimpMesh->mBitangents->x;
            biTangent[i].y = assimpMesh->mBitangents->y;
            biTangent[i].z = assimpMesh->mBitangents->z;
        }
    }

    std::vector<unsigned int> indices;
    for(unsigned int i = 0; i < assimpMesh->mNumFaces; ++i)
    {
        const aiFace& face = assimpMesh->mFaces[i];
        for(unsigned int j = 0; j < face.mNumIndices; ++j)
            indices.push_back(face.mIndices[j]);
    }

    //std::map<TextureTarget, Texture> textures = findTextures(modelPath.parent_path());

    std::vector<VertexShaderAttribute> attributes;  //(5);
    attributes.reserve(5);

    attributes.push_back(VertexShaderAttribute::createVertexShaderAttributeInfo(0, 3, positions));
    attributes.push_back(VertexShaderAttribute::createVertexShaderAttributeInfo(1, 3, normals));
    attributes.push_back(VertexShaderAttribute::createVertexShaderAttributeInfo(2, 2, textureCoords));
    attributes.push_back(VertexShaderAttribute::createVertexShaderAttributeInfo(3, 3, tangent));
    attributes.push_back(VertexShaderAttribute::createVertexShaderAttributeInfo(4, 3, biTangent));

    return std::make_shared<Mesh>(attributes, indices);
}

std::vector<Texture> ResourceLoader::loadTextures(std::filesystem::path& resDirectory)
{
    std::vector<Texture> textures;
    std::vector<std::string> paths;
    for(auto& path_it : std::filesystem::recursive_directory_iterator(resDirectory))
    {
        if(checkExtension(path_it.path().extension().string(), textureExtensions))
        {
            paths.push_back(path_it.path().string());
            continue;
        }

        if(checkExtension(path_it.path().extension().string(), {".hdr"}))
        {
            //ResourceManager::getInstance()->addCubemapTexturePath(path_it.path().string());
        }
    }

    std::vector<std::future<std::pair<std::string, gli::texture>>> futures;
    futures.reserve(paths.size());
    for(const auto& path : paths)
    {
        futures.push_back(std::async(std::launch::async, [&path]() { return loadTextureFromFile(path); }));
    }

    unsigned int texturesLoaded = 0;
    while(texturesLoaded < paths.size())
    {
        for(auto& future : futures)
        {
            if(!future.valid())
                continue;

            const auto [path, texture] = future.get();
            if(const auto optionalResult = loadTexture(path, texture); optionalResult)
            {
                textures.emplace_back(optionalResult.value());
                ++texturesLoaded;
            }
        }
    }
    return textures;
}

std::pair<std::string, gli::texture> ResourceLoader::loadTextureFromFile(const std::string& path)
{
    return {path, gli::load(path)};
}

std::optional<Texture> ResourceLoader::loadTexture(const std::string& path)
{
    int tex_width, tex_height, nr_channels;
    unsigned char* pixels = nullptr;
    pixels = stbi_load(path.c_str(), &tex_width, &tex_height, &nr_channels, 0);

    if(pixels == nullptr)
    {
        std::string error = "Texture from path: " + path + " cannot be loaded!";
        return std::nullopt;
    }

    GLenum format{};
    switch(nr_channels)
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
    }

    GLuint texture;
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, format, tex_width, tex_height, 0, format, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    stbi_image_free(pixels);
    Texture tex{texture, path};
    return tex;
}

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

    utils::createTexture2D(hdrTexture, width, height, GL_RGB16F, GL_RGB, 
        GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, false, data);

    stbi_image_free(data);

    auto tex = std::make_shared<PbrCubemapTexture>(hdrTexture, path, 1024);
    glDeleteTextures(1, &hdrTexture);
    return tex;
}

std::optional<Texture> ResourceLoader::loadTexture(const std::string& path, const gli::texture& texture)
{
    if(texture.empty())
        return std::nullopt;

    gli::gl GL(gli::gl::PROFILE_GL33);
    gli::gl::format const Format = GL.translate(texture.format(), texture.swizzles());
    GLenum Target = GL.translate(texture.target());

    GLuint TextureName = 0;
    glGenTextures(1, &TextureName);
    glBindTexture(Target, TextureName);
    glTexParameteri(Target, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(Target, GL_TEXTURE_MAX_LEVEL, static_cast<GLint>(texture.levels() - 1));
    glTexParameteri(Target, GL_TEXTURE_SWIZZLE_R, Format.Swizzles[0]);
    glTexParameteri(Target, GL_TEXTURE_SWIZZLE_G, Format.Swizzles[1]);
    glTexParameteri(Target, GL_TEXTURE_SWIZZLE_B, Format.Swizzles[2]);
    glTexParameteri(Target, GL_TEXTURE_SWIZZLE_A, Format.Swizzles[3]);
    glTexParameterf(Target, GL_TEXTURE_MAX_ANISOTROPY, Spark::maxAnisotropicFiltering);

    glm::tvec3<GLsizei> const Extent(texture.extent());
    GLsizei const FaceTotal = static_cast<GLsizei>(texture.layers() * texture.faces());

    switch(texture.target())
    {
        case gli::TARGET_1D:
            glTexStorage1D(Target, static_cast<GLint>(texture.levels()), Format.Internal, Extent.x);
            break;
        case gli::TARGET_1D_ARRAY:
        case gli::TARGET_2D:
        case gli::TARGET_CUBE:
            glTexStorage2D(Target, static_cast<GLint>(texture.levels()), Format.Internal, Extent.x,
                           texture.target() == gli::TARGET_2D ? Extent.y : FaceTotal);
            break;
        case gli::TARGET_2D_ARRAY:
        case gli::TARGET_3D:
        case gli::TARGET_CUBE_ARRAY:
            glTexStorage3D(Target, static_cast<GLint>(texture.levels()), Format.Internal, Extent.x, Extent.y,
                           texture.target() == gli::TARGET_3D ? Extent.z : FaceTotal);
            break;
        default:
            assert(0);
            break;
    }

    for(std::size_t Layer = 0; Layer < texture.layers(); ++Layer)
    {
        for(std::size_t Face = 0; Face < texture.faces(); ++Face)
        {
            for(std::size_t Level = 0; Level < texture.levels(); ++Level)
            {
                GLsizei const LayerGL = static_cast<GLsizei>(Layer);
                glm::tvec3<GLsizei> Extent(texture.extent(Level));
                Target = gli::is_target_cube(texture.target()) ? static_cast<GLenum>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + Face) : Target;

                switch(texture.target())
                {
                    case gli::TARGET_1D:
                        if(gli::is_compressed(texture.format()))
                            glCompressedTexSubImage1D(Target, static_cast<GLint>(Level), 0, Extent.x, Format.Internal,
                                                      static_cast<GLsizei>(texture.size(Level)), texture.data(Layer, Face, Level));
                        else
                            glTexSubImage1D(Target, static_cast<GLint>(Level), 0, Extent.x, Format.External, Format.Type,
                                            texture.data(Layer, Face, Level));
                        break;
                    case gli::TARGET_1D_ARRAY:
                    case gli::TARGET_2D:
                    case gli::TARGET_CUBE:
                        if(gli::is_compressed(texture.format()))
                            glCompressedTexSubImage2D(Target, static_cast<GLint>(Level), 0, 0, Extent.x,
                                                      texture.target() == gli::TARGET_1D_ARRAY ? LayerGL : Extent.y, Format.Internal,
                                                      static_cast<GLsizei>(texture.size(Level)), texture.data(Layer, Face, Level));
                        else
                            glTexSubImage2D(Target, static_cast<GLint>(Level), 0, 0, Extent.x,
                                            texture.target() == gli::TARGET_1D_ARRAY ? LayerGL : Extent.y, Format.External, Format.Type,
                                            texture.data(Layer, Face, Level));
                        break;
                    case gli::TARGET_2D_ARRAY:
                    case gli::TARGET_3D:
                    case gli::TARGET_CUBE_ARRAY:
                        if(gli::is_compressed(texture.format()))
                            glCompressedTexSubImage3D(Target, static_cast<GLint>(Level), 0, 0, 0, Extent.x, Extent.y,
                                                      texture.target() == gli::TARGET_3D ? Extent.z : LayerGL, Format.Internal,
                                                      static_cast<GLsizei>(texture.size(Level)), texture.data(Layer, Face, Level));
                        else
                            glTexSubImage3D(Target, static_cast<GLint>(Level), 0, 0, 0, Extent.x, Extent.y,
                                            texture.target() == gli::TARGET_3D ? Extent.z : LayerGL, Format.External, Format.Type,
                                            texture.data(Layer, Face, Level));
                        break;
                    default:
                        assert(0);
                        break;
                }
            }
        }
    }
    Texture tex{TextureName, path};
    return tex;
}
}  // namespace spark
