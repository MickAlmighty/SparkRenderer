#include "ModelLoader.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "utils/CommonUtils.h"
#include "JsonSerializer.h"
#include "Logging.h"
#include "Mesh.h"
#include "Model.h"
#include "Resource.h"
#include "Spark.h"
#include "Texture.h"

namespace spark::loaders
{
std::shared_ptr<resourceManagement::Resource> ModelLoader::load(const std::filesystem::path& resourcesRootPath,
                                                                const std::filesystem::path& resourceRelativePath) const
{
    Assimp::Importer importer;
    const aiScene* scene = nullptr;

    const auto resourcePath = (resourcesRootPath / resourceRelativePath).string();
    scene = importer.ReadFile(resourcePath, aiProcess_Triangulate | aiProcess_CalcTangentSpace);

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
            materialPbr.emplace(spark::TextureTarget::NORMAL_TARGET, std::string(path.data, path.length));
        }

        if(const auto diffuseTexturesCount = material->GetTextureCount(aiTextureType_DIFFUSE); diffuseTexturesCount > 0)
        {
            aiString path{};
            material->GetTexture(aiTextureType_DIFFUSE, 0, &path);
            materialPbr.emplace(spark::TextureTarget::DIFFUSE_TARGET, std::string(path.data, path.length));
        }

        if(const auto metalnessTexturesCount = material->GetTextureCount(aiTextureType_METALNESS); metalnessTexturesCount > 0)
        {
            aiString path{};
            material->GetTexture(aiTextureType_METALNESS, 0, &path);
            materialPbr.emplace(spark::TextureTarget::METALNESS_TARGET, std::string(path.data, path.length));
        }

        if(const auto roughnessTexturesCount = material->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS); roughnessTexturesCount > 0)
        {
            aiString path{};
            material->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &path);
            materialPbr.emplace(spark::TextureTarget::ROUGHNESS_TARGET, std::string(path.data, path.length));
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
        meshes.push_back(loadMesh(scene->mMeshes[i], materials, resourcePath));
    }

    return std::make_shared<resources::Model>(resourceRelativePath, meshes);
}

bool ModelLoader::isExtensionSupported(const std::string& ext) const
{
    const auto supportedExts = supportedExtensions();
    const auto it = std::find_if(supportedExts.begin(), supportedExts.end(), [&ext](const auto& e) { return e == ext; });
    return it != supportedExts.end();
}

std::vector<std::string> ModelLoader::supportedExtensions() const
{
    return {".obj", ".fbx"};
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
            textures.emplace(spark::TextureTarget::DIFFUSE_TARGET, texture);
            continue;
        }

        size = texture_path.path().string().find("_Normal");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture =
                Spark::get().getResourceLibrary().getResourceByFullPath<resources::Texture>(texture_path.path().string());
            textures.emplace(spark::TextureTarget::NORMAL_TARGET, texture);
            continue;
        }

        size = texture_path.path().string().find("_Roughness");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture =
                Spark::get().getResourceLibrary().getResourceByFullPath<resources::Texture>(texture_path.path().string());
            textures.emplace(spark::TextureTarget::ROUGHNESS_TARGET, texture);
            continue;
        }

        size = texture_path.path().string().find("_Metalness");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture =
                Spark::get().getResourceLibrary().getResourceByFullPath<resources::Texture>(texture_path.path().string());
            textures.emplace(spark::TextureTarget::METALNESS_TARGET, texture);
            continue;
        }

        size = texture_path.path().string().find("_Height");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture =
                Spark::get().getResourceLibrary().getResourceByFullPath<resources::Texture>(texture_path.path().string());
            textures.emplace(spark::TextureTarget::HEIGHT_TARGET, texture);
            continue;
        }

        size = texture_path.path().string().find("_AO");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture =
                Spark::get().getResourceLibrary().getResourceByFullPath<resources::Texture>(texture_path.path().string());
            textures.emplace(spark::TextureTarget::AO_TARGET, texture);
        }
    }

    return textures;
}

std::shared_ptr<spark::Mesh> ModelLoader::loadMesh(aiMesh* assimpMesh, std::vector<std::map<spark::TextureTarget, std::string>>& materials,
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

    return std::make_shared<spark::Mesh>(attributes, indices, textures);
}
}  // namespace spark::loaders
