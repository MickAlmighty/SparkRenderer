#ifndef MODEL_MESH_H
#define MODEL_MESH_H

#include <vector>

#include "Component.h"

namespace spark {

class Mesh;
class ModelMesh : public Component
{
public:
	ModelMesh(std::vector<Mesh>& meshes, std::string&& modelName = "ModelMesh");
	ModelMesh() = default;
	~ModelMesh();

	void setModel(std::pair<std::string, std::vector<Mesh>> model);
	
	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;

private:
	std::string modelPath;
	std::vector<Mesh> meshes{};
};

}
#endif