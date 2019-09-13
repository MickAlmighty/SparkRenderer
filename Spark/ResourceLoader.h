#pragma once
#include <iostream>
#include <assimp/scene.h>
#include "Mesh.h"
#include "Model.h"
#include <map>
#include <filesystem>

const std::vector<std::string> modelExtensions = {
	".obj", ".fbx", ".FBX", ".max"
	};

const std::vector<std::string> textureExtensions = {
	".jpg", ".png"
	};

class ResourceLoader
{
private:
	ResourceLoader();
	~ResourceLoader();
	static Model* loadModel(const std::filesystem::path& path);
	static bool checkExtension(std::string&& extension, const std::vector<std::string>& extensions);
	static std::vector<std::unique_ptr<Mesh>> loadMeshes(const aiScene * scene, const std::filesystem::path& modelPath);
	static std::unique_ptr<Mesh> loadMesh(aiMesh* assimpMesh, const std::filesystem::path& modelPath);
	static std::map<TextureTarget, Texture> findTextures(const std::filesystem::path& modelDirectory);
	static Texture loadTexture(std::string&& path);
public:
	static std::map<std::string, Model*> loadModels(std::filesystem::path& modelsDirectory);
	static std::vector<Texture> loadTextures(std::filesystem::path& resDirectory);
};
