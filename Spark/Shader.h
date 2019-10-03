#ifndef SHADER_H
#define SHADER_H

#include <string>

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace spark {

class Shader
{
private:
	GLuint ID;
public:
	Shader(const std::string& vertexShaderPath, const std::string& fragmentShaderPath);
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

}
#endif