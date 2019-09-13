#pragma once
#include "Structs.h"
#include <vector>
#include "Component.h"

class Mesh : public Component
{
private:
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;

	GLuint vao{};
	GLuint vbo{};
	GLuint ebo{};
public:
	Mesh(std::vector<Vertex>& vertices, std::vector<unsigned int> indices, std::string&& newName = "Mesh");
	void update() override;
	void fixedUpdate() override;
	void setup();
	void draw();
	~Mesh();
};

