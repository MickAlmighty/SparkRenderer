#pragma once
#include <map>
#include <GLFW/glfw3.h>
#include "Structs.h"
#include "Model.h"

class ResourceManager
{
	std::vector<Texture> textures;
	std::map<std::string, Model*> models;
public:
	Texture findTexture(const std::string&& path);
	Model* findModel(const std::string&& path);
	ResourceManager();
	~ResourceManager();
	static ResourceManager* getInstance();
	void loadResources();
};

