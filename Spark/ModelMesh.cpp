#include "ModelMesh.h"

ModelMesh::ModelMesh(std::vector<std::unique_ptr<Mesh>>& meshes, std::string&& modelName) : Component(modelName)
{
	this->meshes = std::move(meshes);
}

ModelMesh::~ModelMesh()
{
#ifdef DEBUG
	std::cout << "ModelMesh deleted!" << std::endl;
#endif
}

void ModelMesh::update()
{
}

void ModelMesh::fixedUpdate()
{
}

void ModelMesh::draw()
{
	for(auto& mesh_ptr: meshes)
	{
		mesh_ptr->draw();
	}
}
