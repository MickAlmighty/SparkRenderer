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

	Mesh(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices, std::map<TextureTarget, Texture>& meshTextures, std::string&& newName_ = "Mesh");
	~Mesh() = default;

	void setup();
	void addToRenderQueue(glm::mat4 model);
	void draw(std::shared_ptr<Shader>& shader, glm::mat4 model);
	void cleanup() const;
	

private:
	GLuint vao{};
	GLuint vbo{};
	GLuint ebo{};
};

}
#endif