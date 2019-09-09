#pragma once
#include <vector>
#include "Mesh.h"
#include <memory>
#include "Transform.h"

class Model
{
private:
	std::vector<std::unique_ptr<Mesh>> meshes;
public:
	Model(std::vector<std::unique_ptr<Mesh>>& meshes);
	~Model();

	Transform transform;
	void draw();
};

