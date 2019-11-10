#ifndef MODEL_MESH_H
#define MODEL_MESH_H

#include <vector>

#include "Component.h"

namespace spark {

class Mesh;
class ModelMesh final : public Component
{
public:
	ModelMesh(std::vector<Mesh>& meshes, std::string&& modelName = "ModelMesh");
    ModelMesh(const ModelMesh&) = delete;
    ModelMesh(const ModelMesh&&) = delete;
    ModelMesh& operator=(const ModelMesh&) = delete;
    ModelMesh& operator=(const ModelMesh&&) = delete;

	void setModel(std::pair<std::string, std::vector<Mesh>> model);
	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;
    std::string getModelPath() const;
    void setModelPath(const std::string modelPath);
	ModelMesh();
    private:
	std::string modelPath;
	std::vector<Mesh> meshes{};
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(Component);
};

}
#endif