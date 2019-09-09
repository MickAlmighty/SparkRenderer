#pragma once
#include <iostream>
#include <assimp/scene.h>
#include "Mesh.h"
#include "Model.h"

class ResourceLoader
{
private:
	ResourceLoader();
	~ResourceLoader();
	static std::unique_ptr<Mesh> loadMesh(aiMesh* assimpMesh);
public:
	static Model* loadModel(std::string&& path);
	
};

