#include "Mesh.h"

#include <iostream>

#include "EngineSystems/SparkRenderer.h"
#include "Shader.h"
#include "Logging.h"

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

	glCreateBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), reinterpret_cast<const void*>(vertices.data()), GL_STATIC_DRAW);

	glCreateBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), reinterpret_cast<const void*>(indices.data()), GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void*>(offsetof(Vertex, pos)));

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void*>(offsetof(Vertex, normal)));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void*>(offsetof(Vertex, texCoords)));

	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void*>(offsetof(Vertex, tangent)));

	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void*>(offsetof(Vertex, bitangent)));

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

	for (auto& texture_it : textures)
	{
		glBindTextureUnit(static_cast<GLuint>(texture_it.first), texture_it.second.ID);
	}

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

    SPARK_TRACE("Mesh deleted!");
}

}
