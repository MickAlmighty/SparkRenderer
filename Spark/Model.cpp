#include "Model.h"

Model::Model(std::vector<std::unique_ptr<Mesh>>& meshes, std::map<TextureTarget, Texture>& textures)
{
	this->meshes = std::move(meshes);
	this->textures = std::move(textures);
}

Model::~Model()
{

}

void Model::draw()
{
	for(auto& texture_it : textures)
	{
		glActiveTexture(GL_TEXTURE0 + texture_it.first);
		glBindTexture(GL_TEXTURE_2D, texture_it.second.ID);
	}

	for(auto& mesh_ptr: meshes)
	{
		mesh_ptr->draw();
	}

	glBindTexture(GL_TEXTURE_2D, 0);
}
