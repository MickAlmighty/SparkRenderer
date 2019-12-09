#ifndef MESH_H
#define MESH_H

#include <vector>
#include <map>

#include "Enums.h"
#include "Structs.h"

namespace spark {

class Shader;
class Mesh
{
public:
	ShaderType shaderType = ShaderType::DEFAULT_SHADER;
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;
	std::map<TextureTarget, Texture> textures;
	glm::mat4 model;
	
	GLuint vao{};
	GLuint vbo{};
	GLuint ebo{};
	
	Mesh(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices, std::map<TextureTarget, Texture>& meshTextures, std::string&& newName_ = "Mesh");
	~Mesh() = default;
	bool operator==(const Mesh& mesh) const;
	bool operator<(const Mesh& mesh) const;

	void setup();
	void addToRenderQueue(const glm::mat4& model);
	void draw(std::shared_ptr<Shader>& shader, glm::mat4 model);
	void draw(std::shared_ptr<Shader>& shader);
	void bindTextures() const;
	void cleanup() const;
};

}
#endif