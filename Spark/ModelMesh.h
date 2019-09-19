#pragma once
#include <vector>
#include <Mesh.h>
#include <memory>
#include <Enums.h>
#include <map>

class ModelMesh : public Component
{
private:
	std::string modelPath;
	std::vector<Mesh> meshes{};
public:
	ModelMesh(std::vector<Mesh>& meshes, std::string&& modelName = "ModelMesh");
	ModelMesh();
	void setModel(std::pair<std::string, std::vector<Mesh>> model);
	~ModelMesh();
	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;
	SerializableType getSerializableType() override;
	Json::Value serialize() override;
	void deserialize(Json::Value& root) override;
};

