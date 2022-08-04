#include "GltfLoader.hpp"

#include <sstream>

#include <stb_image/stb_image.h>
#include <stb_image/stb_image_write.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Enums.h"
#include "Logging.h"
#include "Mesh.h"
#include "Model.h"
#include "ResourceIdentifier.h"
#include "Spark.h"
#include "Texture.h"
#include "tiny_gltf_wrapper.h"

namespace
{
enum class ComponentType
{
    BYTE = 5120,
    UNSIGNED_BYTE = 5121,
    SHORT = 5122,
    UNSIGNED_SHORT = 5123,
    UNSIGNED_INT = 5125,
    FLOAT = 5126
};

template<typename T>
std::vector<T> collectData(const tinygltf::Model& model, int accessorIdx)
{
    const tinygltf::Accessor& accessor = model.accessors[accessorIdx];
    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];

    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

    const auto* positionArray = reinterpret_cast<const T*>(buffer.data.data() + bufferView.byteOffset + accessor.byteOffset);

    std::vector<T> outputBuffer(accessor.count);
    for(size_t i = 0; i < accessor.count; ++i)
    {
        outputBuffer[i] = positionArray[i];
    }

    return outputBuffer;
}

auto splitMetallicRoughnessTexture(const std::filesystem::path& resourcesRootPath, const std::filesystem::path& metallicRoughnessTexPath,
                                   const std::filesystem::path& metallicTexPath, const std::filesystem::path& roughnessTexName)
{
    const auto inputPath = (resourcesRootPath / metallicRoughnessTexPath).string();
    int width{0}, height{0}, channels{0};
    unsigned char* pixels = stbi_load(inputPath.c_str(), &width, &height, &channels, 0);

    if(pixels == nullptr)
    {
        SPARK_ERROR("Texture from path: {} cannot be loaded!", inputPath);
    }
    else
    {
        std::vector<unsigned char> metallicData(width * height * channels);
        std::vector<unsigned char> roughnessData(width * height * channels);

        for(int i = 0; i < width * height * channels; i += channels)
        {
            roughnessData[i] = pixels[i + 1];
            metallicData[i] = pixels[i + 2];
        }

        const auto outputChannels = 3;
        if(metallicTexPath.extension() == ".jpg")
        {
            stbi_write_jpg((resourcesRootPath / metallicTexPath).string().c_str(), width, height, outputChannels, metallicData.data(), 100);
        }
        else if(metallicTexPath.extension() == ".png")
        {
            stbi_write_png((resourcesRootPath / metallicTexPath).string().c_str(), width, height, outputChannels, metallicData.data(), width);
        }

        if(roughnessTexName.extension() == ".jpg")
        {
            stbi_write_jpg((resourcesRootPath / roughnessTexName).string().c_str(), width, height, outputChannels, roughnessData.data(), 100);
        }
        else if(roughnessTexName.extension() == ".png")
        {
            stbi_write_png((resourcesRootPath / roughnessTexName).string().c_str(), width, height, outputChannels, roughnessData.data(), width);
        }
    }

    stbi_image_free(pixels);
}

std::filesystem::path getRelativeCachedTexturePath(const std::filesystem::path& relativePath, const std::string& texName,
                                                   const std::string& extension)
{
    const auto textureName = texName + extension;

    const auto texPath = relativePath / textureName;
    std::stringstream cachedFilename;
    cachedFilename << std::filesystem::hash_value(texPath) << texPath.extension().string();

    return std::filesystem::path("cache") / cachedFilename.str();
}

auto collectMaterials(const tinygltf::Model& model, const std::filesystem::path& resourcesRootPath, const std::filesystem::path& resourceRelativePath)
{
    std::vector<std::map<spark::TextureTarget, std::filesystem::path>> materials;

    if(model.materials.empty())
    {
        return materials;
    }

    const auto resourceFullPath = resourcesRootPath / resourceRelativePath;
    materials.reserve(model.materials.size());
    for(const auto& material : model.materials)
    {
        std::map<spark::TextureTarget, std::filesystem::path> pbrMaterial{};
        const std::filesystem::path assetDirectoryPath = resourceFullPath.parent_path();

        if(const auto& normalTextureInfo = material.normalTexture; normalTextureInfo.index != -1)
        {
            pbrMaterial.emplace(spark::TextureTarget::NORMAL_TARGET, assetDirectoryPath / model.images[normalTextureInfo.index].uri);
        }

        const auto& pbrMaterialInfo = material.pbrMetallicRoughness;
        if(pbrMaterialInfo.baseColorTexture.index != -1)
        {
            pbrMaterial.emplace(spark::TextureTarget::DIFFUSE_TARGET, assetDirectoryPath / model.images[pbrMaterialInfo.baseColorTexture.index].uri);
        }

        if(pbrMaterialInfo.metallicRoughnessTexture.index != -1)
        {
            const auto& metallicRoughnessTex = std::filesystem::path(model.images[pbrMaterialInfo.metallicRoughnessTexture.index].uri);
            const auto nameWithoutExtension = metallicRoughnessTex.stem().string();
            const auto extension = metallicRoughnessTex.extension().string();

            const auto relativeCachedMetallnessTexturePath =
                getRelativeCachedTexturePath(assetDirectoryPath, nameWithoutExtension + "_metallness", extension);
            const auto relativeCachedRoughnessTexturePath =
                getRelativeCachedTexturePath(assetDirectoryPath, nameWithoutExtension + "_roughness", extension);

            if(const bool atLeastOneFileDoesNotExist = !std::filesystem::exists(resourcesRootPath / relativeCachedMetallnessTexturePath) ||
                                                       !std::filesystem::exists(resourcesRootPath / relativeCachedRoughnessTexturePath);
               atLeastOneFileDoesNotExist)
            {
                const auto relativeMetallicRoughnessTexPath = assetDirectoryPath / metallicRoughnessTex;
                splitMetallicRoughnessTexture(resourcesRootPath, relativeMetallicRoughnessTexPath, relativeCachedMetallnessTexturePath,
                                              relativeCachedRoughnessTexturePath);
            }

            pbrMaterial.emplace(spark::TextureTarget::METALNESS_TARGET, relativeCachedMetallnessTexturePath);
            pbrMaterial.emplace(spark::TextureTarget::ROUGHNESS_TARGET, relativeCachedRoughnessTexturePath);
        }

        materials.push_back(pbrMaterial);
    }

    return materials;
}

auto collectMaterialTextures(const std::filesystem::path& assetDirectoryPath,
                             const std::vector<std::map<spark::TextureTarget, std::filesystem::path>>& materials, int materialIndex)
{
    std::map<spark::TextureTarget, std::shared_ptr<spark::resources::Texture>> textures;
    if(const auto& material = materials[materialIndex]; !material.empty())
    {
        for(const auto& [target, localTexPath] : material)
        {
            if(const auto& texture =
                   spark::Spark::get().getResourceLibrary().getResourceByRelativePath<spark::resources::Texture>(localTexPath.string());
               texture)
            {
                textures.emplace(target, texture);
            }
            else
            {
                SPARK_INFO("texture {} could not be loaded", localTexPath.string());
            }
        }
    }

    return textures;
}

auto collectAttributes(const tinygltf::Model& model, const tinygltf::Primitive& primitive)
{
    std::vector<glm::vec3> positions{};
    std::vector<glm::vec3> normals{};
    std::vector<glm::vec2> textureCoords{};
    std::vector<glm::vec3> tangents{};

    if(const auto it = primitive.attributes.find("POSITION"); it != primitive.attributes.end())
    {
        positions = collectData<glm::vec3>(model, it->second);
    }

    if(const auto it = primitive.attributes.find("NORMAL"); it != primitive.attributes.end())
    {
        normals = collectData<glm::vec3>(model, it->second);
    }

    if(const auto it = primitive.attributes.find("TEXCOORD_0"); it != primitive.attributes.end())
    {
        textureCoords = collectData<glm::vec2>(model, it->second);
    }

    if(const auto it = primitive.attributes.find("TANGENT"); it != primitive.attributes.end())
    {
        const auto tangentsBuffer = collectData<glm::vec4>(model, it->second);

        tangents.reserve(tangentsBuffer.size());
        for(auto& tangent : tangentsBuffer)
        {
            tangents.emplace_back(tangent);
        }
    }
    else
    {
        //#todo calculate vector on surface on a triangle plane and then calculate tangent
        for(size_t i = 0; i < normals.size(); ++i)
        {
            glm::vec3 position = glm::normalize(positions[i]);
            glm::vec3 normal = glm::normalize(normals[i]);
            glm::vec3 tangent = glm::cross(position, normal);

            tangents.push_back(tangent);
        }
    }

    return std::vector{spark::VertexAttribute(0, 3, positions), spark::VertexAttribute(1, 3, normals), spark::VertexAttribute(2, 2, textureCoords),
                       spark::VertexAttribute(3, 3, tangents)};
}

template<typename T>
std::vector<unsigned int> convertIndicesToUInt32Array(T* indicesArray, size_t indicesCount)
{
    std::vector<unsigned int> indices(indicesCount);

    for(size_t i = 0; i < indicesCount; ++i)
    {
        indices[i] = indicesArray[i];
    }

    return indices;
}

std::vector<unsigned int> collectIndices(const tinygltf::Model& model, const tinygltf::Primitive& primitive)
{
    if(const auto accessorIdxToIndicesBuffer = primitive.indices; accessorIdxToIndicesBuffer != -1)
    {
        const tinygltf::Accessor& accessor = model.accessors[accessorIdxToIndicesBuffer];
        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];

        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
        const auto* indices = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;

        if(accessor.componentType == static_cast<int>(ComponentType::UNSIGNED_BYTE))
        {
            return convertIndicesToUInt32Array(indices, accessor.count);
        }
        if(accessor.componentType == static_cast<int>(ComponentType::UNSIGNED_SHORT))
        {
            return convertIndicesToUInt32Array(reinterpret_cast<const unsigned short*>(indices), accessor.count);
        }
        if(accessor.componentType == static_cast<int>(ComponentType::UNSIGNED_INT))
        {
            return convertIndicesToUInt32Array(reinterpret_cast<const unsigned int*>(indices), accessor.count);
        }

        return {};
    }

    return {};
}
}  // namespace

namespace spark::loaders
{
std::shared_ptr<resourceManagement::Resource> GltfLoader::load(const std::filesystem::path& resourcesRootPath,
                                                               const std::filesystem::path& resourceRelativePath) const
{
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err;
    std::string warn;

    const auto path = resourcesRootPath / resourceRelativePath;
    const bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, path.string());

    if(!warn.empty())
    {
        SPARK_WARN("Warn: %s", warn.c_str());
    }

    if(!err.empty())
    {
        SPARK_ERROR("Err: %s", err.c_str());
    }

    if(!ret)
    {
        SPARK_ERROR("Failed to parse glTF\n");
        return nullptr;
    }

    const auto materials = collectMaterials(model, resourcesRootPath, resourceRelativePath);

    std::vector<std::shared_ptr<Mesh>> meshes;
    for(const auto& mesh : model.meshes)
    {
        for(const auto& primitive : mesh.primitives)
        {
            auto attributes = collectAttributes(model, primitive);
            auto indices = collectIndices(model, primitive);
            auto textures = collectMaterialTextures(path.parent_path(), materials, primitive.material);

            meshes.push_back(std::make_shared<Mesh>(attributes, indices, textures));
        }
    }

    return std::make_shared<resources::Model>(resourceRelativePath, meshes);
}

bool GltfLoader::isExtensionSupported(const std::string& ext) const
{
    const auto supportedExts = supportedExtensions();
    const auto it = std::find_if(supportedExts.begin(), supportedExts.end(), [&ext](const auto& e) { return e == ext; });
    return it != supportedExts.end();
}

std::vector<std::string> GltfLoader::supportedExtensions() const
{
    return {".gltf"};
}
}  // namespace spark::loaders
