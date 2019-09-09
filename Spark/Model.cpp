#include "Model.h"

Model::Model(std::vector<std::unique_ptr<Mesh>>& meshes)
{
	this->meshes = std::move(meshes);
}

Model::~Model()
{

}

void Model::draw()
{
	for(auto& mesh_ptr: meshes)
	{
		mesh_ptr->draw();
	}
}
