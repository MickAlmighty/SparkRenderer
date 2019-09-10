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
	".jpg"
	};

class ResourceLoader
{
private:
	ResourceLoader();
	~ResourceLoader();
	static std::unique_ptr<Mesh> loadMesh(aiMesh* assimpMesh);
	static std::vector<std::unique_ptr<Mesh>> loadMeshes(const aiScene * scene);
	static bool checkExtension(std::string&& extension, const std::vector<std::string>& extensions);
	static std::map<TextureTarget, Texture> findTextures(const std::filesystem::path& modelDirectory);
public:
	static Model* loadModel(const std::filesystem::path& path);
	static std::map<std::string, Model*> loadModels(std::filesystem::path&& modelsDirectory);

	static std::vector<Texture> loadTextures(std::filesystem::path&& resDirectory);
	static Texture loadTexture(std::string&& path);
	
};

