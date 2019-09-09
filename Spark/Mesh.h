#pragma once
#include "Structs.h"
#include <vector>

class Mesh
{
private:
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;

	GLuint vao{};
	GLuint vbo{};
	GLuint ebo{};
public:
	Mesh(std::vector<Vertex>& vertices, std::vector<unsigned int> indices);
	void setup();
	void draw();
	~Mesh();
};

