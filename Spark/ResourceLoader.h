#ifndef RESOURCE_LOADER_H
#define RESOURCE_LOADER_H

#include <filesystem>
#include <map>

#include "Enums.h"

struct aiMesh;
struct aiScene;
namespace gli {
	class texture;
}

namespace spark {

class MeshModel;
class Mesh;
struct Texture;

const std::vector<std::string> ModelMeshExtensions = {
	".obj", ".fbx", ".FBX", ".max"
};

const std::vector<std::string> textureExtensions = {
	".DDS", ".KTX"//".jpg", ".png"
};

class ResourceLoader
{
public:
	static std::map<std::string, std::vector<Mesh>> loadModels(std::filesystem::path& modelDirectory);
	static std::vector<Texture> loadTextures(std::filesystem::path& resDirectory);

	ResourceLoader(const ResourceLoader& rhs) = delete;
	ResourceLoader& operator=(const ResourceLoader& rhs) = delete;

private:
	ResourceLoader() = default;
	~ResourceLoader() = default;

	static std::vector<Mesh> loadModel(const std::filesystem::path& path);
	static bool checkExtension(std::string&& extension, const std::vector<std::string>& extensions);
	static std::vector<Mesh> loadMeshes(const aiScene * scene, const std::filesystem::path& modelPath);
	static Mesh loadMesh(aiMesh* assimpMesh, const std::filesystem::path& modelPath);
	static std::map<TextureTarget, Texture> findTextures(const std::filesystem::path& modelDirectory);
	static void loadTextureFromFile(std::vector<std::pair<std::string, gli::texture>>& loadedFiles, const std::string& path);
	static std::pair<std::string, gli::texture> loadTextureFromFile(const std::string& path);
	static Texture loadTexture(const std::string& path);
	static Texture loadTexture(const std::string& path, const gli::texture& texture);
};

}

#endif