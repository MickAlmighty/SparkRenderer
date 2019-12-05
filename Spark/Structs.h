#ifndef STRUCTS_H
#define STRUCTS_H

#include <vector>
#include <filesystem>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <json/value.h>

#include "LocalTranform.h"
#include "Timer.h"
#include "WorldTransform.h"

namespace spark {

	struct Transform
	{
		LocalTransform local;
		WorldTransform world;
	};

	struct Uniform
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

	struct InitializationVariables
	{
		unsigned int width;
		unsigned int height;
		bool vsync = false;
		std::filesystem::path pathToModels;
		std::filesystem::path pathToResources;

		Json::Value serialize() const
		{
			Json::Value root;
			root["width"] = width;
			root["height"] = height;
			root["pathToModels"] = pathToModels.string();
			root["pathToResources"] = pathToResources.string();
			root["vsync"] = vsync;
			return root;
		}

		void deserialize(Json::Value& root)
		{
			width = root.get("width", 1280).asInt();
			height = root.get("height", 720).asInt();
			pathToModels = root["pathToModels"].asString();
			pathToResources = root["pathToResources"].asString();
			vsync = root.get("vsync", false).asBool();
		}
	};

	struct Texture
	{
		GLuint ID;
		std::string path;
	};

	struct PbrCubemapTexture
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

	struct Cube
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

	typedef  struct {
		GLuint  count;
		GLuint  instanceCount;
		GLuint  firstIndex;
		GLuint  baseVertex;
		GLuint  baseInstance;
	} DrawElementsIndirectCommand;

	struct MeshInfo
	{
		unsigned int verticesSize{ 0 };
		unsigned int elementsCount{ 0 };
		unsigned int verticesOffset{ 0 };
		unsigned int indicesOffset{ 0 };
	};

	struct MultiDrawIndirectBuffer
	{
		GLuint vao{};
		GLuint vbo{};
		GLuint ebo{};

		unsigned int meshCounter{ 0 };

		std::vector<glm::vec3> vertices;
		std::vector<GLuint> indices;

		GLuint lastVertexOffset{ 0 };
		GLuint lastIndexOffset{ 0 };
		std::vector<MeshInfo> meshInfos;

		MultiDrawIndirectBuffer()
		{
			glGenVertexArrays(1, &vao);
			glGenBuffers(1, &vbo);
			glGenBuffers(1, &ebo);

			const GLuint vertexBindingPoint = 0;

			glBindVertexArray(vao);
			
			glVertexAttribFormat(0, 3, GL_FLOAT, false, 0);
			glVertexAttribBinding(0, vertexBindingPoint);
			glEnableVertexAttribArray(0);
			glBindVertexBuffer(vertexBindingPoint, vbo, 0, sizeof(glm::vec3));

			glBindVertexArray(0);
		}

		void clear() const
		{
			glDeleteBuffers(1, &vbo);
			glDeleteBuffers(1, &ebo);
			glDeleteVertexArrays(1, &vao);
		}

		void addMesh(const std::vector<glm::vec3>& vertices_, const std::vector<GLuint>& indices_)
		{
			vertices.insert(vertices.end(), vertices_.begin(), vertices_.end());
			indices.insert(indices.end(), indices_.begin(), indices_.end());

			MeshInfo info;
			info.verticesSize = static_cast<GLuint>(vertices_.size());
			info.elementsCount = static_cast<GLuint>(indices_.size());
			info.verticesOffset = lastVertexOffset;
			info.indicesOffset = lastIndexOffset;
			meshInfos.push_back(info);

			lastVertexOffset += static_cast<GLuint>(vertices_.size());
			lastIndexOffset += static_cast<GLuint>(indices_.size());
			++meshCounter;
		}

		void draw()
		{
			//Timer pathRenderingTimer("Paths rendering");
			if (meshCounter == 0)
				return;

			glBindVertexArray(vao);

			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STREAM_DRAW);

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STREAM_DRAW);

			std::vector<DrawElementsIndirectCommand> commands(meshCounter);

			for(unsigned int i = 0; i < meshCounter; ++i)
			{
				commands[i].count = meshInfos[i].elementsCount;
				commands[i].instanceCount = 1;
				commands[i].firstIndex = meshInfos[i].indicesOffset;
				commands[i].baseVertex = meshInfos[i].verticesOffset;
				commands[i].baseInstance = 0;
			}

			GLuint indirectBuffer{0};
			glGenBuffers(1, &indirectBuffer);
			glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirectBuffer);
			glBufferData(GL_DRAW_INDIRECT_BUFFER, commands.size() * sizeof(DrawElementsIndirectCommand), commands.data(), GL_STREAM_COPY);

			glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 0, static_cast<GLsizei>(commands.size()), 0);

			glDeleteBuffers(1, &indirectBuffer);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			glBindVertexArray(0);
		}
	};

	struct DirectionalLightData
	{
		alignas(16) glm::vec3 direction;
		alignas(16) glm::vec3 color;
	};

	struct PointLightData
	{
		alignas(16) glm::vec3 position;
		alignas(16) glm::vec3 color;
	};

	struct SpotLightData
	{
		alignas(16) glm::vec3 position;
		float cutOff;
		glm::vec3 color;
		float outerCutOff;
		glm::vec3 direction;
	};

}
#endif