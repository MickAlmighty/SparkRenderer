#pragma once
#include <vector>
#include "Mesh.h"
#include <memory>
#include "Transform.h"
#include "Enums.h"
#include <map>

class Model
{
private:
	std::vector<std::unique_ptr<Mesh>> meshes;
	std::map<TextureTarget, Texture> textures;
public:
	Model(std::vector<std::unique_ptr<Mesh>>& meshes, std::map<TextureTarget, Texture>& textures);
	~Model();

	Transform transform;
	void draw();
};

