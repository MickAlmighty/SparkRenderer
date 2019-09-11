#pragma once
#include <map>
#include <GLFW/glfw3.h>
#include "Structs.h"
#include "Model.h"
#include "Enums.h"
#include "Shader.h"

class ResourceManager
{
	std::vector<Texture> textures;
	std::map<std::string, Model*> models;
	std::map<ShaderType, std::shared_ptr<Shader>> shaders;
public:
	Texture findTexture(const std::string&& path);
	Model* findModel(const std::string&& path);
	std::shared_ptr<Shader>& getShader(ShaderType type);
	ResourceManager();
	~ResourceManager();
	static ResourceManager* getInstance();
	void loadResources();
};

