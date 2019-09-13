#include "Model.h"

Model::Model(std::vector<std::unique_ptr<Mesh>>& meshes)
{
	this->meshes = std::move(meshes);
}

Model::~Model()
{
#ifdef DEBUG
	std::cout << "Model deleted!" << std::endl;
#endif
}

void Model::draw()
{
	for(auto& mesh_ptr: meshes)
	{
		mesh_ptr->draw();
	}
}
