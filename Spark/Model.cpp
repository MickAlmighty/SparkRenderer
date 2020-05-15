#include "Model.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "Logging.h"
#include "ResourceLibrary.h"
#include "Spark.h"
#include "Texture.h"
#include "Timer.h"

namespace spark::resources
{
Model::Model(const resourceManagement::ResourceIdentifier& identifier) : Resource(identifier) {}

bool Model::isResourceReady()
{
    return isLoadedIntoDeviceMemory() && isLoadedIntoRAM();
}

bool Model::load()
{
    // SPARK_INFO("Model: {}. Loading meshes from file.", getName());
    meshes = loadModel(id.getFullPath());

    setLoadedIntoRam(true);
    return true;
}

bool Model::unload()
{
    // SPARK_INFO("Model: {}. Freeing meshes from RAM.", getName());
    meshes.clear();

    setLoadedIntoRam(false);
    return true;
}

bool Model::gpuLoad()
{
    // Timer timer("Loading mesh on GPU for model: " + id.getResourceName().string());
    for(const auto& mesh : meshes)
    {
        mesh->gpuLoad();
    }

    setLoadedIntoDeviceMemory(true);
    return true;
}

bool Model::gpuUnload()
{
    // SPARK_INFO("Model: {}. Freeing meshes data from gpu memory.", getName());
    for(const auto& mesh : meshes)
    {
        mesh->gpuUnload();
    }

    setLoadedIntoDeviceMemory(false);
    return true;
}

std::vector<std::shared_ptr<Mesh>> Model::getMeshes() const
{
    return meshes;
}

std::vector<std::shared_ptr<Mesh>> Model::loadModel(const std::filesystem::path& path)
{
    // Timer timer("loadModel( " + path.string() + " )");
    Assimp::Importer importer;
    const aiScene* scene = nullptr;

    scene = importer.ReadFile(path.string(), aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        SPARK_ERROR("{}", importer.GetErrorString());
        throw std::exception(importer.GetErrorString());
    }

    std::vector<std::shared_ptr<Mesh>> meshes;
    for(unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        meshes.push_back(loadMesh(scene->mMeshes[i], path));
    }
    return meshes;
}

std::map<TextureTarget, std::shared_ptr<resources::Texture>> findTextures(const std::filesystem::path& modelDirectory)
{
    std::map<TextureTarget, std::shared_ptr<resources::Texture>> textures;
    for(auto& texture_path : std::filesystem::recursive_directory_iterator(modelDirectory))
    {
        size_t size = texture_path.path().string().find("_Diffuse.DDS");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture = Spark::getResourceLibrary()->getResourceByPath<resources::Texture>(texture_path.path().string());
            textures.emplace(TextureTarget::DIFFUSE_TARGET, texture);
            continue;
        }

        size = texture_path.path().string().find("_Normal.DDS");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture = Spark::getResourceLibrary()->getResourceByPath<resources::Texture>(texture_path.path().string());
            textures.emplace(TextureTarget::NORMAL_TARGET, texture);
        }

        size = texture_path.path().string().find("_Roughness.DDS");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture = Spark::getResourceLibrary()->getResourceByPath<resources::Texture>(texture_path.path().string());
            textures.emplace(TextureTarget::ROUGHNESS_TARGET, texture);
        }

        size = texture_path.path().string().find("_Metalness.DDS");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture = Spark::getResourceLibrary()->getResourceByPath<resources::Texture>(texture_path.path().string());
            textures.emplace(TextureTarget::METALNESS_TARGET, texture);
        }

        size = texture_path.path().string().find("_Height.DDS");
        if(size != std::string::npos)
        {
            std::shared_ptr<resources::Texture> texture = Spark::getResourceLibrary()->getResourceByPath<resources::Texture>(texture_path.path().string());
            textures.emplace(TextureTarget::HEIGHT_TARGET, texture);
        }
    }

    return textures;
}

std::shared_ptr<Mesh> Model::loadMesh(aiMesh* assimpMesh, const std::filesystem::path& modelPath)
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

    std::map<TextureTarget, std::shared_ptr<resources::Texture>> textures = findTextures(id.getFullPath().parent_path());

    std::vector<VertexShaderAttribute> attributes;  //(5);
    attributes.reserve(5);

    attributes.push_back(VertexShaderAttribute::createVertexShaderAttributeInfo(0, 3, positions));
    attributes.push_back(VertexShaderAttribute::createVertexShaderAttributeInfo(1, 3, normals));
    attributes.push_back(VertexShaderAttribute::createVertexShaderAttributeInfo(2, 2, textureCoords));
    attributes.push_back(VertexShaderAttribute::createVertexShaderAttributeInfo(3, 3, tangent));
    attributes.push_back(VertexShaderAttribute::createVertexShaderAttributeInfo(4, 3, biTangent));

    return std::make_shared<Mesh>(attributes, indices, textures);
}
}  // namespace spark::resources
