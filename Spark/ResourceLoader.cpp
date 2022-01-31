#include "ResourceLoader.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <gli/load.hpp>
#include <stb_image/stb_image.h>

#include "utils/CommonUtils.h"
#include "JsonSerializer.h"
#include "Logging.h"
#include "Mesh.h"
#include "Model.h"
#include "Spark.h"
#include "Texture.h"
#include "Shader.h"

namespace spark
{
using Path = std::filesystem::path;

std::shared_ptr<resourceManagement::Resource> ResourceLoader::createHdrTexture(
    const std::shared_ptr<resourceManagement::ResourceIdentifier>& resourceIdentifier)
{
    int width, height, nrComponents;
    float* data = stbi_loadf(resourceIdentifier->getFullPath().string().c_str(), &width, &height, &nrComponents, 0);

    if(!data)
    {
        SPARK_ERROR("Failed to load HDR image '{}'.", resourceIdentifier->getFullPath().string());
        return nullptr;
    }

    auto hdrTexture = utils::createTexture2D(width, height, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, false, data);

    stbi_image_free(data);
    stbi_image_free(data);

    return std::make_shared<resources::Texture>(resourceIdentifier->getRelativePath(), std::move(hdrTexture), width, height);
}

std::shared_ptr<resourceManagement::Resource> ResourceLoader::createCompressedTexture(
    const std::shared_ptr<resourceManagement::ResourceIdentifier>& resourceIdentifier)
{
    const auto gliTexture = gli::load(resourceIdentifier->getFullPath().string());

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

    return std::make_shared<resources::Texture>(resourceIdentifier->getRelativePath(), utils::UniqueTextureHandle(textureID), Extent.x, Extent.y);
}

std::shared_ptr<resourceManagement::Resource> ResourceLoader::createUncompressedTexture(
    const std::shared_ptr<resourceManagement::ResourceIdentifier>& resourceIdentifier)
{
    int width{0}, height{0}, channels{0};
    unsigned char* pixels = stbi_load(resourceIdentifier->getFullPath().string().c_str(), &width, &height, &channels, 0);

    if(pixels == nullptr)
    {
        SPARK_ERROR("Texture from path: " + resourceIdentifier->getFullPath().string() + " cannot be loaded!");
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

    GLuint texId;
    glCreateTextures(GL_TEXTURE_2D, 1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    float maxAnisotropicFiltering{1.0f};
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAnisotropicFiltering);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, maxAnisotropicFiltering);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    stbi_image_free(pixels);

    return std::make_shared<resources::Texture>(resourceIdentifier->getRelativePath(), utils::UniqueTextureHandle(texId), width, height);
}

std::shared_ptr<resourceManagement::Resource> ResourceLoader::createModel(
    const std::shared_ptr<resourceManagement::ResourceIdentifier>& resourceIdentifier)
{
    Assimp::Importer importer;
    const aiScene* scene = nullptr;

    scene = importer.ReadFile(resourceIdentifier->getFullPath().string(), aiProcess_Triangulate | aiProcess_CalcTangentSpace);

    if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        SPARK_ERROR("{}", importer.GetErrorString());
        return nullptr;
    }

    std::vector<std::map<TextureTarget, std::string>> materials;
    for(int i = 0; i < scene->mNumMaterials; ++i)
    {
        const aiMaterial* material = scene->mMaterials[i];
        std::map<TextureTarget, std::string> materialPbr{};

        if(const auto normalTexturesCount = material->GetTextureCount(aiTextureType_NORMALS); normalTexturesCount > 0)
        {
            aiString path{};
            material->GetTexture(aiTextureType_NORMALS, 0, &path);
            materialPbr.emplace(TextureTarget::NORMAL_TARGET, std::string(path.data, path.length));
        }

        if(const auto diffuseTexturesCount = material->GetTextureCount(aiTextureType_DIFFUSE); diffuseTexturesCount > 0)
        {
            aiString path{};
            material->GetTexture(aiTextureType_DIFFUSE, 0, &path);
            materialPbr.emplace(TextureTarget::DIFFUSE_TARGET, std::string(path.data, path.length));
        }

        if(const auto metalnessTexturesCount = material->GetTextureCount(aiTextureType_METALNESS); metalnessTexturesCount > 0)
        {
            aiString path{};
            material->GetTexture(aiTextureType_METALNESS, 0, &path);
            materialPbr.emplace(TextureTarget::METALNESS_TARGET, std::string(path.data, path.length));
        }

        if(const auto roughnessTexturesCount = material->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS); roughnessTexturesCount > 0)
        {
            aiString path{};
            material->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &path);
            materialPbr.emplace(TextureTarget::ROUGHNESS_TARGET, std::string(path.data, path.length));
        }

        if(const auto unknownTypeTexturesCount = material->GetTextureCount(aiTextureType_UNKNOWN); unknownTypeTexturesCount > 0)
        {
            /*aiString path{};
            material->GetTexture(aiTextureType_UNKNOWN, 0, &path);
            materialPbr.emplace(TextureTarget::AO_TARGET, std::string(path.data, path.length));*/
        }

        materials.push_back(materialPbr);
    }

    std::vector<std::shared_ptr<Mesh>> meshes;
    for(unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        meshes.push_back(loadMesh(scene->mMeshes[i], materials, resourceIdentifier->getFullPath()));
    }

    return std::make_shared<resources::Model>(resourceIdentifier->getRelativePath(), meshes);
}

std::shared_ptr<resourceManagement::Resource> ResourceLoader::createShader(
    const std::shared_ptr<resourceManagement::ResourceIdentifier>& resourceIdentifier)
{
    return std::make_shared<resources::Shader>(resourceIdentifier->getFullPath());
}

std::map<TextureTarget, std::shared_ptr<resources::Texture>> findTextures(const std::filesystem::path& modelDirectory)
{
    std::map<TextureTarget, std::shared_ptr<resources::Texture>> textures;
    for(auto& texture_path : std::filesystem::recursive_directory_iterator(modelDirectory))
    {
        size_t size = texture_path.path().string().find("_Diffuse");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture =
                Spark::get().getResourceLibrary().getResourceByFullPath<resources::Texture>(texture_path.path().string());
            textures.emplace(TextureTarget::DIFFUSE_TARGET, texture);
            continue;
        }

        size = texture_path.path().string().find("_Normal");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture =
                Spark::get().getResourceLibrary().getResourceByFullPath<resources::Texture>(texture_path.path().string());
            textures.emplace(TextureTarget::NORMAL_TARGET, texture);
            continue;
        }

        size = texture_path.path().string().find("_Roughness");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture =
                Spark::get().getResourceLibrary().getResourceByFullPath<resources::Texture>(texture_path.path().string());
            textures.emplace(TextureTarget::ROUGHNESS_TARGET, texture);
            continue;
        }

        size = texture_path.path().string().find("_Metalness");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture =
                Spark::get().getResourceLibrary().getResourceByFullPath<resources::Texture>(texture_path.path().string());
            textures.emplace(TextureTarget::METALNESS_TARGET, texture);
            continue;
        }

        size = texture_path.path().string().find("_Height");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture =
                Spark::get().getResourceLibrary().getResourceByFullPath<resources::Texture>(texture_path.path().string());
            textures.emplace(TextureTarget::HEIGHT_TARGET, texture);
            continue;
        }

        size = texture_path.path().string().find("_AO");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture =
                Spark::get().getResourceLibrary().getResourceByFullPath<resources::Texture>(texture_path.path().string());
            textures.emplace(TextureTarget::AO_TARGET, texture);
        }
    }

    return textures;
}

std::shared_ptr<Mesh> ResourceLoader::loadMesh(aiMesh* assimpMesh, std::vector<std::map<TextureTarget, std::string>>& materials,
                                               const std::filesystem::path& path)
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
            tangent[i].x = assimpMesh->mTangents[i].x;
            tangent[i].y = assimpMesh->mTangents[i].y;
            tangent[i].z = assimpMesh->mTangents[i].z;

            biTangent[i].x = assimpMesh->mBitangents[i].x;
            biTangent[i].y = assimpMesh->mBitangents[i].y;
            biTangent[i].z = assimpMesh->mBitangents[i].z;
        }
    }

    std::vector<unsigned int> indices;
    for(unsigned int i = 0; i < assimpMesh->mNumFaces; ++i)
    {
        const aiFace& face = assimpMesh->mFaces[i];
        for(unsigned int j = 0; j < face.mNumIndices; ++j)
            indices.push_back(face.mIndices[j]);
    }

    std::map<TextureTarget, std::shared_ptr<resources::Texture>> textures;
    if(const auto& materialsPbr = materials.at(assimpMesh->mMaterialIndex); !materialsPbr.empty())
    {
        for(const auto& [target, localTexPath] : materialsPbr)
        {
            const auto pathToTexture = (path.parent_path() / localTexPath).string();
            textures.emplace(target, Spark::get().getResourceLibrary().getResourceByFullPath<resources::Texture>(pathToTexture));
        }
    }
    else
    {
        textures = findTextures(path.parent_path());
    }

    std::vector<VertexAttribute> attributes{{0, 3, positions}, {1, 3, normals}, {2, 2, textureCoords}, {3, 3, tangent}, {4, 3, biTangent}};

    return std::make_shared<Mesh>(attributes, indices, textures);
}

std::shared_ptr<resourceManagement::Resource> ResourceLoader::createScene(
    const std::shared_ptr<resourceManagement::ResourceIdentifier>& resourceIdentifier)
{
    if(auto scene = JsonSerializer().loadSceneFromFile(resourceIdentifier->getFullPath()); scene)
    {
        scene->setPath(resourceIdentifier->getRelativePath());
        return scene;
    }

    return nullptr;
}

std::shared_ptr<resourceManagement::Resource> ResourceLoader::createAnimation(
    const std::shared_ptr<resourceManagement::ResourceIdentifier>& resourceIdentifier)
{
    if(const auto animation = JsonSerializer().loadShared<resources::AnimationData>(resourceIdentifier->getFullPath()); animation)
    {
        animation->setPath(resourceIdentifier->getRelativePath());
        return animation;
    }

    return nullptr;
}
}  // namespace spark
