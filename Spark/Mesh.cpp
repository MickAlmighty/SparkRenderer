#include "Mesh.h"

#include <iostream>

#include "EngineSystems/SparkRenderer.h"
#include "Shader.h"

namespace spark {

Mesh::Mesh(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices, std::map<TextureTarget, Texture>& meshTextures, std::string&& newName_)
{
	this->vertices = std::move(vertices);
	this->indices = std::move(indices);
	this->textures = std::move(meshTextures);

	setup();
}

void Mesh::setup()
{
	glCreateVertexArrays(1, &vao);
	glBindVertexArray(vao);

	const GLuint vertexBindingPoint = 0;

	//this attribute layout is connected to vao, not to vbo by glVertexAttribPointer
	glVertexAttribFormat(0, 3, GL_FLOAT, false, offsetof(Vertex, pos));
	glVertexAttribBinding(0, vertexBindingPoint);
	glEnableVertexAttribArray(0);

	glVertexAttribFormat(1, 3, GL_FLOAT, false, offsetof(Vertex, normal));
	glVertexAttribBinding(1, vertexBindingPoint);
	glEnableVertexAttribArray(1);

	glVertexAttribFormat(2, 2, GL_FLOAT, false, offsetof(Vertex, texCoords));
	glVertexAttribBinding(2, vertexBindingPoint);
	glEnableVertexAttribArray(2);

	glVertexAttribFormat(3, 3, GL_FLOAT, false, offsetof(Vertex, tangent));
	glVertexAttribBinding(3, vertexBindingPoint);
	glEnableVertexAttribArray(3);

	glVertexAttribFormat(4, 4, GL_FLOAT, false, offsetof(Vertex, bitangent));
	glVertexAttribBinding(4, vertexBindingPoint);
	glEnableVertexAttribArray(4);

	glCreateBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), reinterpret_cast<const void*>(vertices.data()), GL_STATIC_DRAW);
	glBindVertexBuffer(vertexBindingPoint, vbo, 0, sizeof(Vertex));

	glCreateBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), reinterpret_cast<const void*>(indices.data()), GL_STATIC_DRAW);

	glBindVertexArray(0);
}

void Mesh::addToRenderQueue(glm::mat4 model)
{
	auto f = [this, model](std::shared_ptr<Shader>& shader)
	{
		draw(shader, model);
	};
	SparkRenderer::getInstance()->renderQueue[shaderType].push_back(f);
}


void Mesh::draw(std::shared_ptr<Shader>& shader, glm::mat4 model)
{
	shader->setMat4("model", model);

	std::array<GLuint, 4> textureIDs{};
	for (auto& texture_it : textures)
	{
		//glBindTextureUnit(static_cast<GLuint>(texture_it.first), texture_it.second.ID);
		textureIDs[static_cast<GLuint>(texture_it.first) - 1] = texture_it.second.ID;
	}

	glBindTextures(1, 4, textureIDs.data());

	glBindVertexArray(vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	glBindTextures(static_cast<GLuint>(TextureTarget::DIFFUSE_TARGET), static_cast<GLsizei>(textures.size()), nullptr);
}

void Mesh::cleanup() const
{
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &ebo);
	glDeleteVertexArrays(1, &vao);

#ifdef DEBUG
	std::cout << "Mesh deleted!" << std::endl;
#endif
}

}