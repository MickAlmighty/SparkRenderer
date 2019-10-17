#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include <map>

#include "Structs.h"
#include "Enums.h"

namespace spark {

class Mesh;
class Shader;
class ResourceManager
{
public:
	static ResourceManager* getInstance();

	void addTexture(Texture tex);
	Texture findTexture(const std::string&& path) const;
	std::vector<Mesh> findModelMeshes(const std::string& path) const;
	std::vector<std::string> getPathsToModels() const;
	std::shared_ptr<Shader> getShader(const ShaderType& type) const;
	std::shared_ptr<Shader> getShader(const std::string& name) const;
	std::vector<std::string> getShaderNames() const;
	std::vector<Texture> getTextures() const;
	void loadResources();
	void cleanup();

private:
	std::vector<Texture> textures;
	std::map<std::string, std::vector<Mesh>> models;
	std::map<ShaderType, std::shared_ptr<Shader>> shaders;

	ResourceManager() = default;
	~ResourceManager() = default;
};
}
#endif