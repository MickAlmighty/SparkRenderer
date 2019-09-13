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
		glBindTextureUnit(static_cast<GLuint>(texture_it.first), texture_it.second.ID);
	}

	for(auto& mesh_ptr: meshes)
	{
		mesh_ptr->draw();
	}

	glBindTextures(static_cast<GLuint>(DIFFUSE_TARGET), textures.size(), nullptr);
}
