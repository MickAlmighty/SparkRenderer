#pragma once
#include <map>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Structs.h>
#include <ModelMesh.h>
#include <Enums.h>
#include <Shader.h>

class ResourceManager
{
	std::vector<Texture> textures;
	std::map<std::string, std::vector<Mesh>> models;
	std::map<ShaderType, std::shared_ptr<Shader>> shaders;
	ResourceManager() = default;
	~ResourceManager() = default;
public:
	Texture findTexture(const std::string&& path);
	std::vector<Mesh> findModelMeshes(const std::string& path);
	std::vector<std::string> getPathsToModels();
	std::shared_ptr<Shader>& getShader(ShaderType type);

	static ResourceManager* getInstance();
	void loadResources();
	void cleanup();
};

