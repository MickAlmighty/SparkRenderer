#pragma once
#include <Structs.h>
#include <vector>
#include <Component.h>
#include <map>
#include <Enums.h>
#include <Shader.h>

class Mesh
{
private:
	ShaderType shaderType = DEFAULT_SHADER;
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;
	std::map<TextureTarget, Texture> textures;

	GLuint vao{};
	GLuint vbo{};
	GLuint ebo{};
public:
	Mesh(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices, std::map<TextureTarget, Texture>& meshTextures, std::string&& newName = "Mesh");
	void setup();
	void addToRenderQueue(glm::mat4 model);
	void draw(std::shared_ptr<Shader>& shader, glm::mat4 model);
	void cleanup();
	~Mesh();
};

