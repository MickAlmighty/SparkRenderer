#ifndef SHADER_H
#define SHADER_H

#include <map>
#include <string>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <list>
#include <vector>

namespace spark {

struct Uniform;
class Shader
{
public:
	std::string name;

	static GLenum shaderTypeFromString(const std::string& type);

	Shader(const std::string& vertexShaderPath, const std::string& fragmentShaderPath);
	Shader(const std::string& shaderPath);
	~Shader();
	std::string loadShader(const std::string& shaderPath);
	std::map<GLenum, std::string> preProcess(const std::string& shaderPath);
	std::vector<GLuint> compileShaders(std::map<GLenum, std::string> shaders) const;
	std::list<Uniform> gatherUniforms(std::stringstream&& stream) const;
	void linkProgram(const std::vector<GLuint>& ids);
	void queryUniformLocations(const std::list<Uniform>& uniforms);

	void use() const;

	GLuint getLocation(const std::string& name) const;
	void setBool(const std::string &name, bool value) const;
	void setInt(const std::string &name, int value) const;
	void setFloat(const std::string &name, float value) const;
	void setVec2(const std::string &name, glm::vec2 value) const;
	void setVec3(const std::string &name, glm::vec3 value) const;
	void setMat4(const std::string &name, glm::mat4 value) const;
private:
	GLuint ID {0};
	std::map<Uniform, GLuint> uniformLocations;
};

}
#endif