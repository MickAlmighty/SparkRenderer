#pragma once
#include <glad/glad.h>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

class Shader
{
private:
	GLuint ID;
public:
	Shader(const char* vertexShaderPath, const char* fragmentShaderPath);
	~Shader();
	GLuint compileVertexShader(const char* vertexShaderSource);
	GLuint compileFragmentShader(const char* fragmentShaderSource);
	void linkProgram(GLuint vertexShader, GLuint fragmentShader);
	
	void use();

	void setBool(const std::string &name, bool value) const;
	void setInt(const std::string &name, int value) const;
	void setFloat(const std::string &name, float value) const;
	void setVec2(const std::string &name, glm::vec2 value) const;
	void setVec3(const std::string &name, glm::vec3 value) const;
	void setMat4(const std::string &name, glm::mat4 value) const;
};

