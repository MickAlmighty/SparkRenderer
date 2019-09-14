#pragma once
#include <vector>
#include "Mesh.h"
#include <memory>
#include "Transform.h"
#include "Enums.h"
#include <map>

class ModelMesh : public Component
{
private:
	std::vector<std::unique_ptr<Mesh>> meshes;
public:
	ModelMesh(std::vector<std::unique_ptr<Mesh>>& meshes, std::string&& modelName = "ModelMesh");
	~ModelMesh();
	void update() override;
	void fixedUpdate() override;
	Transform transform;
	void draw();
};

