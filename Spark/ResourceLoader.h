#pragma once
#include <iostream>
#include <assimp/scene.h>
#include <Mesh.h>
#include <ModelMesh.h>
#include <map>
#include <filesystem>

const std::vector<std::string> ModelMeshExtensions = {
	".obj", ".fbx", ".FBX", ".max"
	};

const std::vector<std::string> textureExtensions = {
	".jpg", ".png"
	};

class ResourceLoader
{
private:
	ResourceLoader() = default;
	~ResourceLoader() = default;
	static std::vector<Mesh> loadModel(const std::filesystem::path& path);
	static bool checkExtension(std::string&& extension, const std::vector<std::string>& extensions);
	static std::vector<Mesh> loadMeshes(const aiScene * scene, const std::filesystem::path& modelPath);
	static Mesh loadMesh(aiMesh* assimpMesh, const std::filesystem::path& modelPath);
	static std::map<TextureTarget, Texture> findTextures(const std::filesystem::path& modelDirectory);
	static Texture loadTexture(std::string&& path);
public:
	static std::map<std::string, std::vector<Mesh>> loadModels(std::filesystem::path& modelDirectory);
	static std::vector<Texture> loadTextures(std::filesystem::path& resDirectory);
};
