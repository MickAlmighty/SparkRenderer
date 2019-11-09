#ifndef MODEL_MESH_H
#define MODEL_MESH_H

#include <vector>

#include "Component.h"

namespace spark {

class Mesh;
class ModelMesh final : public Component
{
public:
	ModelMesh() = default;
	ModelMesh(std::vector<Mesh>& meshes, std::string&& modelName = "ModelMesh");
    ModelMesh(const ModelMesh&) = delete;
    ModelMesh(const ModelMesh&&) = delete;
    ModelMesh& operator=(const ModelMesh&) = delete;
    ModelMesh& operator=(const ModelMesh&&) = delete;

	void setModel(std::pair<std::string, std::vector<Mesh>> model);
	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;
private:
	std::string modelPath;
	std::vector<Mesh> meshes{}; //TODO: add proper mesh loading for deserialization
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(Component);
};

}
#endif