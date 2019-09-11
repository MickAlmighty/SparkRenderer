#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <vector>
#include <filesystem>

struct InitializationVariables
{
	unsigned int width;
	unsigned int height;
	std::filesystem::path pathToModels;
	std::filesystem::path pathToResources;
};

struct Texture
{
	GLuint ID;
	std::string path;
};

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 texCoords;
	glm::vec3 tangent;
	glm::vec3 bitangent;
};

struct QuadVertex
{
	glm::vec3 pos;
	glm::vec2 texCoords;
};


struct ScreenQuad
{
	GLuint vao{};
	GLuint vbo{};

	std::vector<QuadVertex> vertices =
	{
		{{-1.0f, 1.0f, 0.0f},	{0.0f, 1.0f}},
		{{1.0f, 1.0f, 0.0f},	{1.0f, 1.0f }},
		{{1.0f, -1.0f, 0.0f},	{1.0f, 0.0f}},

		{{-1.0f, 1.0f, 0.0f},	{0.0f, 1.0f}},
		{{1.0f, -1.0f, 0.0f},	{1.0f, 0.0f}},
		{{-1.0f, -1.0f, 0.0f},	{0.0f, 0.0f}}
	};

	void setup()
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(QuadVertex), &vertices[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), reinterpret_cast<void*>(offsetof(QuadVertex, pos)));

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), reinterpret_cast<void*>(offsetof(QuadVertex, texCoords)));

		glBindVertexArray(0);
	}

	void draw()
	{
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, vertices.size());
		glBindVertexArray(0);
	}



	~ScreenQuad()
	{
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};
