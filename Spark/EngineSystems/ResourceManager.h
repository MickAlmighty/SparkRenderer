#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include <map>
#include <Structs.h>
#include <Enums.h>

class Mesh;
class Shader;
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
	std::vector<Texture> getTextures();

	static ResourceManager* getInstance();
	void loadResources();
	void cleanup();
};

#endif