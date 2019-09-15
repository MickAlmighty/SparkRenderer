#pragma once
#include <vector>
#include <Mesh.h>
#include <memory>
#include <Enums.h>
#include <map>

class ModelMesh : public Component
{
private:
	std::vector<Mesh> meshes;
public:
	ModelMesh(std::vector<Mesh>& meshes, std::string&& modelName = "ModelMesh");
	ModelMesh(const ModelMesh* modelMesh);
	~ModelMesh();
	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;
};

