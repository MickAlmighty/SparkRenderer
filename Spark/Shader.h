#ifndef SHADER_H
#define SHADER_H

#include <map>
#include <string>

#include <glad/glad.h>
#include <glm/glm.hpp>


namespace spark {

class Shader
{
public:
	Shader(const std::string& vertexShaderPath, const std::string& fragmentShaderPath);
	~Shader();
	GLuint compileVertexShader(const char* vertexShaderSource);
	GLuint compileFragmentShader(const char* fragmentShaderSource);
	void linkProgram(GLuint vertexShader, GLuint fragmentShader);

	void use();

	GLuint getLocation(const std::string& name) const;
	void setBool(const std::string &name, bool value) const;
	void setInt(const std::string &name, int value) const;
	void setFloat(const std::string &name, float value) const;
	void setVec2(const std::string &name, glm::vec2 value) const;
	void setVec3(const std::string &name, glm::vec3 value) const;
	void setMat4(const std::string &name, glm::mat4 value) const;
private:
	GLuint ID {0};
	mutable std::map<std::string, GLuint> uniformLocations;
};

}
#endif