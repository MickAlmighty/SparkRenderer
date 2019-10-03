#ifndef MODEL_MESH_H
#define MODEL_MESH_H

#include <vector>

#include "Component.h"

namespace spark {

class Mesh;
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

}
#endif