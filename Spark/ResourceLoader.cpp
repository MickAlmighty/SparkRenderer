#include <ResourceLoader.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <stb_image/stb_image.h>
#include <Structs.h>
#include <filesystem>
#include <EngineSystems/ResourceManager.h>

using Path = std::filesystem::path;

std::map<std::string, std::vector<Mesh>> ResourceLoader::loadModels(std::filesystem::path& modelDirectory)
{
	std::map<std::string, std::vector<Mesh>> models;
	for (auto& path_it : std::filesystem::recursive_directory_iterator(modelDirectory))
	{
		if(checkExtension(path_it.path().extension().string(), ModelMeshExtensions))
		{
			models.emplace(path_it.path().string(), loadModel(path_it.path()));
		}
	}

	return models;
}

std::vector<Mesh> ResourceLoader::loadModel(const Path& path)
{
	Assimp::Importer importer;
	const aiScene *scene = importer.ReadFile(path.string(), aiProcessPreset_TargetRealtime_Fast | aiProcess_FlipUVs);

	if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		throw std::exception(importer.GetErrorString());
	}

	std::vector<Mesh> meshes = loadMeshes(scene, path);

	return meshes;
}

std::vector<Mesh> ResourceLoader::loadMeshes(const aiScene* scene, const std::filesystem::path& modelPath)
{
	std::vector<Mesh> meshes;
	for (unsigned int i = 0; i < scene->mNumMeshes; i++)
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
	for (auto& texture_path : std::filesystem::recursive_directory_iterator(modelDirectory))
	{
		size_t size = texture_path.path().string().find("_Diffuse");
		if( size != std::string::npos)
		{
			Texture tex = ResourceManager::getInstance()->findTexture(texture_path.path().string());
			textures.emplace(DIFFUSE_TARGET, tex);
			continue;
		}

		size = texture_path.path().string().find("_Normal");
		if (size != std::string::npos)
		{
			Texture tex = ResourceManager::getInstance()->findTexture(texture_path.path().string());
			textures.emplace(NORMAL_TARGET, tex);
		}
	}

	return textures;
}

Mesh ResourceLoader::loadMesh(aiMesh* assimpMesh, const std::filesystem::path& modelPath)
{
	std::vector<Vertex> vertices(assimpMesh->mNumVertices);

	for (unsigned int i = 0; i < assimpMesh->mNumVertices; i++)
	{
		Vertex v{};

		v.pos.x = assimpMesh->mVertices[i].x;
		v.pos.y = assimpMesh->mVertices[i].y;
		v.pos.z = assimpMesh->mVertices[i].z;

		if (assimpMesh->HasNormals())
		{
			v.normal.x = assimpMesh->mNormals[i].x;
			v.normal.y = assimpMesh->mNormals[i].y;
			v.normal.z = assimpMesh->mNormals[i].z;
		}

		if (assimpMesh->HasTextureCoords(0))
		{
			v.texCoords.x = assimpMesh->mTextureCoords[0][i].x;
			v.texCoords.y = assimpMesh->mTextureCoords[0][i].y;
		}
		else
			v.texCoords = glm::vec2(0.0f, 0.0f);

		vertices[i] = v;
	}

	std::vector<unsigned int> indices;
	for (unsigned int i = 0; i < assimpMesh->mNumFaces; i++)
	{
		aiFace face = assimpMesh->mFaces[i];
		for (unsigned int j = 0; j < face.mNumIndices; j++)
			indices.push_back(face.mIndices[j]);
	}

	std::map<TextureTarget, Texture> textures = findTextures(modelPath.parent_path());

	return Mesh(vertices, indices, textures);
}

std::vector<Texture> ResourceLoader::loadTextures(std::filesystem::path& resDirectory)
{
	std::vector<Texture> textures;
	for (auto& path_it : std::filesystem::recursive_directory_iterator(resDirectory))
	{
		if (checkExtension(path_it.path().extension().string(), textureExtensions))
		{
			textures.push_back(loadTexture(path_it.path().string()));
		}
	}
	return textures;
}

Texture ResourceLoader::loadTexture(std::string&& path)
{
	int tex_width, tex_height, nr_channels;
	unsigned char* pixels = nullptr;
	pixels = stbi_load(path.c_str(), &tex_width, &tex_height, &nr_channels, 0);

	if(pixels == nullptr )
	{
		std::string error = "Texture from path: " + path + " cannot be loaded!";
		throw std::exception(error.c_str());
	}

	GLenum format{};
	switch(nr_channels)
	{
		case(1): format = GL_RED; break;
		case(2): format = GL_RG; break;
		case(3): format = GL_RGB; break;
		case(4): format = GL_RGBA; break;
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

	return {texture, path};
}
