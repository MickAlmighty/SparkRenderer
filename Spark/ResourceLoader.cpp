#include "ResourceLoader.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <gli/gli.hpp>
#include <stb_image/stb_image.h>

#include "EngineSystems/ResourceManager.h"
#include "Mesh.h"
#include "Structs.h"
#include <future>

namespace spark {

	using Path = std::filesystem::path;

	std::map<std::string, std::vector<Mesh>> ResourceLoader::loadModels(std::filesystem::path& modelDirectory)
	{
		std::map<std::string, std::vector<Mesh>> models;
		for (auto& path_it : std::filesystem::recursive_directory_iterator(modelDirectory))
		{
			if (checkExtension(path_it.path().extension().string(), ModelMeshExtensions))
			{
				models.emplace(path_it.path().string(), loadModel(path_it.path()));
			}
		}

		return models;
	}

	std::vector<Mesh> ResourceLoader::loadModel(const Path& path)
	{
		Assimp::Importer importer;
		const aiScene *scene = importer.ReadFile(path.string(), aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
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
			if (size != std::string::npos)
			{
				Texture tex = ResourceManager::getInstance()->findTexture(texture_path.path().string());
				textures.emplace(TextureTarget::DIFFUSE_TARGET, tex);
				continue;
			}

			size = texture_path.path().string().find("_Normal");
			if (size != std::string::npos)
			{
				Texture tex = ResourceManager::getInstance()->findTexture(texture_path.path().string());
				textures.emplace(TextureTarget::NORMAL_TARGET, tex);
			}
			
			size = texture_path.path().string().find("_Roughness");
			if (size != std::string::npos)
			{
				Texture tex = ResourceManager::getInstance()->findTexture(texture_path.path().string());
				textures.emplace(TextureTarget::ROUGHNESS_TARGET, tex);
			}
			
			size = texture_path.path().string().find("_Metalness");
			if (size != std::string::npos)
			{
				Texture tex = ResourceManager::getInstance()->findTexture(texture_path.path().string());
				textures.emplace(TextureTarget::METALNESS_TARGET, tex);
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

			if (assimpMesh->HasTangentsAndBitangents())
			{
				v.tangent.x = assimpMesh->mTangents->x;
				v.tangent.y = assimpMesh->mTangents->y;
				v.tangent.z = assimpMesh->mTangents->z;

				v.bitangent.x = assimpMesh->mBitangents->x;
				v.bitangent.y = assimpMesh->mBitangents->y;
				v.bitangent.z = assimpMesh->mBitangents->z;
			}
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
		std::vector<std::string> paths;
		for (auto& path_it : std::filesystem::recursive_directory_iterator(resDirectory))
		{
			if (checkExtension(path_it.path().extension().string(), textureExtensions))
			{
				//textures.push_back(loadTexture(path_it.path().string()));
				paths.push_back(path_it.path().string());
			}
		}

		std::vector<std::future<std::pair<std::string, gli::texture>>> futures;
		futures.reserve(paths.size());
		for (const auto& path : paths)
		{
			futures.push_back(std::async(std::launch::async, [&path]()
			{
				return loadTextureFromFile(path);
			}));
		}

		std::vector<Texture> textures;
		textures.reserve(paths.size());
		
		unsigned int texturesLoaded = 0;
		while (texturesLoaded < paths.size())
		{
			for (auto& future : futures)
			{
				if (!future.valid())
					continue;

				const auto[path, texture] = future.get();
				textures.emplace_back(loadTexture(path, texture));
				++texturesLoaded;
			}
		}
		return textures;
	}

	void ResourceLoader::loadTextureFromFile(std::vector<std::pair<std::string, gli::texture>>& loadedFiles, const std::string& path)
	{
		static std::mutex m;
		gli::texture t = gli::load(path);
		std::lock_guard<std::mutex> lock(m);
		loadedFiles.emplace_back(path, t);
	}

	std::pair<std::string, gli::texture> ResourceLoader::loadTextureFromFile(const std::string& path)
	{
		return { path, gli::load(path) };
	}

	Texture ResourceLoader::loadTexture(const std::string& path)
	{
		
		int tex_width, tex_height, nr_channels;
		unsigned char* pixels = nullptr;
		pixels = stbi_load(path.c_str(), &tex_width, &tex_height, &nr_channels, 0);

		if (pixels == nullptr)
		{
			std::string error = "Texture from path: " + path + " cannot be loaded!";
			throw std::exception(error.c_str());
		}

		GLenum format{};
		switch (nr_channels)
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
		//glCompressedTexImage2D(GL_TEXTURE_2D, 0, format, tex_width, tex_height, 0, 0, pixels);
		glTexImage2D(GL_TEXTURE_2D, 0, format, tex_width, tex_height, 0, format, GL_UNSIGNED_BYTE, pixels);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);

		stbi_image_free(pixels);

		return { texture, path };
	}

	Texture ResourceLoader::loadTexture(const std::string& path, const gli::texture& texture)
	{
		if (texture.empty())
			return { 0, path };

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

		glm::tvec3<GLsizei> const Extent(texture.extent());
		GLsizei const FaceTotal = static_cast<GLsizei>(texture.layers() * texture.faces());

		switch (texture.target())
		{
		case gli::TARGET_1D:
			glTexStorage1D(
				Target, static_cast<GLint>(texture.levels()), Format.Internal, Extent.x);
			break;
		case gli::TARGET_1D_ARRAY:
		case gli::TARGET_2D:
		case gli::TARGET_CUBE:
			glTexStorage2D(
				Target, static_cast<GLint>(texture.levels()), Format.Internal,
				Extent.x, texture.target() == gli::TARGET_2D ? Extent.y : FaceTotal);
			break;
		case gli::TARGET_2D_ARRAY:
		case gli::TARGET_3D:
		case gli::TARGET_CUBE_ARRAY:
			glTexStorage3D(
				Target, static_cast<GLint>(texture.levels()), Format.Internal,
				Extent.x, Extent.y,
				texture.target() == gli::TARGET_3D ? Extent.z : FaceTotal);
			break;
		default:
			assert(0);
			break;
		}

		for (std::size_t Layer = 0; Layer < texture.layers(); ++Layer)
		{
			for (std::size_t Face = 0; Face < texture.faces(); ++Face)
			{
				for (std::size_t Level = 0; Level < texture.levels(); ++Level)
				{
					GLsizei const LayerGL = static_cast<GLsizei>(Layer);
					glm::tvec3<GLsizei> Extent(texture.extent(Level));
					Target = gli::is_target_cube(texture.target())
						? static_cast<GLenum>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + Face)
						: Target;

					switch (texture.target())
					{
					case gli::TARGET_1D:
						if (gli::is_compressed(texture.format()))
							glCompressedTexSubImage1D(
								Target, static_cast<GLint>(Level), 0, Extent.x,
								Format.Internal, static_cast<GLsizei>(texture.size(Level)),
								texture.data(Layer, Face, Level));
						else
							glTexSubImage1D(
								Target, static_cast<GLint>(Level), 0, Extent.x,
								Format.External, Format.Type,
								texture.data(Layer, Face, Level));
						break;
					case gli::TARGET_1D_ARRAY:
					case gli::TARGET_2D:
					case gli::TARGET_CUBE:
						if (gli::is_compressed(texture.format()))
							glCompressedTexSubImage2D(
								Target, static_cast<GLint>(Level),
								0, 0,
								Extent.x,
								texture.target() == gli::TARGET_1D_ARRAY ? LayerGL : Extent.y,
								Format.Internal, static_cast<GLsizei>(texture.size(Level)),
								texture.data(Layer, Face, Level));
						else
							glTexSubImage2D(
								Target, static_cast<GLint>(Level),
								0, 0,
								Extent.x,
								texture.target() == gli::TARGET_1D_ARRAY ? LayerGL : Extent.y,
								Format.External, Format.Type,
								texture.data(Layer, Face, Level));
						break;
					case gli::TARGET_2D_ARRAY:
					case gli::TARGET_3D:
					case gli::TARGET_CUBE_ARRAY:
						if (gli::is_compressed(texture.format()))
							glCompressedTexSubImage3D(
								Target, static_cast<GLint>(Level),
								0, 0, 0,
								Extent.x, Extent.y,
								texture.target() == gli::TARGET_3D ? Extent.z : LayerGL,
								Format.Internal, static_cast<GLsizei>(texture.size(Level)),
								texture.data(Layer, Face, Level));
						else
							glTexSubImage3D(
								Target, static_cast<GLint>(Level),
								0, 0, 0,
								Extent.x, Extent.y,
								texture.target() == gli::TARGET_3D ? Extent.z : LayerGL,
								Format.External, Format.Type,
								texture.data(Layer, Face, Level));
						break;
					default: assert(0); break;
					}
				}
			}
		}
		return { TextureName, path };
	}
}
