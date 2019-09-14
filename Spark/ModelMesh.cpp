#include <ModelMesh.h>

ModelMesh::ModelMesh(std::vector<Mesh>& meshes, std::string&& modelName) : Component(modelName)
{
	this->meshes = std::move(meshes);
}

ModelMesh::ModelMesh(const ModelMesh* modelMesh)
{
}

ModelMesh::~ModelMesh()
{
#ifdef DEBUG
	std::cout << "ModelMesh deleted!" << std::endl;
#endif
}

void ModelMesh::update()
{
	glm::mat4 model = getGameObject()->transform.world.getMatrix();
	for(Mesh& mesh: meshes)
	{
		mesh.addToRenderQueue(model);
	}
}

void ModelMesh::fixedUpdate()
{
}

void ModelMesh::draw()
{
	/*for(auto& mesh_ptr: meshes)
	{
		mesh_ptr->draw();
	}*/
}
