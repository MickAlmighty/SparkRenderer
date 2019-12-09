#ifndef MODEL_MESH_H
#define MODEL_MESH_H

#include <vector>

#include "Component.h"

namespace spark {

class Mesh;
class ModelMesh : public Component
{
public:
	bool instanced{ false };
	std::vector<Mesh> meshes{};

	ModelMesh(std::vector<Mesh>& meshes, std::string&& modelName = "ModelMesh");
	ModelMesh() = default;
	~ModelMesh();

	void setModel(const std::pair<std::string, std::vector<Mesh>>& model);
	
	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;
	SerializableType getSerializableType() override;
	Json::Value serialize() override;
	void deserialize(Json::Value& root) override;

private:
	std::string modelPath;
};

}
#endif