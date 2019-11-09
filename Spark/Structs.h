#ifndef STRUCTS_H
#define STRUCTS_H

#include "LocalTransform.h"
#include "WorldTransform.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <rttr/registration>

#include <vector>
#include <filesystem>

namespace spark {

	struct Transform final
	{
		LocalTransform local;
		WorldTransform world;
        RTTR_ENABLE();
	};

	struct Uniform final
	{
		std::string type;
		std::string name;
		bool operator!=(const Uniform& rhs) const
		{
			return this->name != rhs.name || this->type != rhs.type;
		}
		bool operator<(const Uniform& rhs) const
		{
			return this->name < rhs.name;
		}
	};

	struct InitializationVariables final
	{
		unsigned int width;
		unsigned int height;
		std::string pathToModels;
		std::string pathToResources;
        RTTR_ENABLE();
	};

	struct Texture
	{
		GLuint ID;
		std::string path;
	};

	struct PbrCubemapTexture final
	{
		GLuint cubemap{};
		GLuint irradianceCubemap{}; 
		GLuint prefilteredCubemap{};
		GLuint brdfLUTTexture{};

		PbrCubemapTexture(GLuint hdrTexture, unsigned int size = 1024);
		~PbrCubemapTexture();

	private:
		void setup(GLuint hdrTexture, unsigned int size);
		GLuint generateCubemap(unsigned int texSize, bool mipmaps = false) const;
		void generateCubemapMipMaps();
	};


	struct Vertex final
	{
		glm::vec3 pos;
		glm::vec3 normal;
		glm::vec2 texCoords;
		glm::vec3 tangent;
		glm::vec3 bitangent;
	};

	struct QuadVertex final
	{
		glm::vec3 pos;
		glm::vec2 texCoords;
	};


	struct ScreenQuad final
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

		void draw() const
		{
			glBindVertexArray(vao);
			glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
			glBindVertexArray(0);
		}

		~ScreenQuad()
		{
			glDeleteBuffers(1, &vbo);
			glDeleteVertexArrays(1, &vao);
		}
	};

	struct Cube final
	{
		GLuint vao{};
		GLuint vbo{};

		float vertices[288] = {
			// back face
			-1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
			 1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
			 1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right         
			 1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
			-1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
			-1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
			// front face
			-1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
			 1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
			 1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
			 1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
			-1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
			-1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
			// left face
			-1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			-1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
			-1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
			-1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
			-1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
			-1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			// right face
			 1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
			 1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
			 1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
			 1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
			 1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
			 1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
			// bottom face
			-1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
			 1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
			 1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
			 1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
			-1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
			-1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
			// top face
			-1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
			 1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			 1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
			 1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			-1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
			-1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left        
		};

		Cube()
		{
			glGenVertexArrays(1, &vao);
			glGenBuffers(1, &vbo);
			// fill buffer
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
			// link vertex attributes
			glBindVertexArray(vao);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);
		}

		void draw() const
		{
			glBindVertexArray(vao);
			glDrawArrays(GL_TRIANGLES, 0, 36);
			glBindVertexArray(0);
		}
	};

	struct DirectionalLightData final
	{
		alignas(16) glm::vec3 direction;
		alignas(16) glm::vec3 color;
	};

	struct PointLightData final
	{
		alignas(16) glm::vec3 position;
		alignas(16) glm::vec3 color;
	};

	struct SpotLightData final
	{
		alignas(16) glm::vec3 position;
		float cutOff;
		glm::vec3 color;
		float outerCutOff;
		glm::vec3 direction;
	};

}
#endif