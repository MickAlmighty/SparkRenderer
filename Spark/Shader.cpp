#include "Shader.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <glm/gtc/type_ptr.hpp>

#include "Structs.h"

namespace spark {

Shader::Shader(const std::string& vertexShaderPath, const std::string& fragmentShaderPath)
{
	// 1. pobierz kod Ÿród³owy Vertex/Fragment Shadera z filePath  
	std::list<Uniform> uniforms;
	std::string vertexCode, fragmentCode;
	std::stringstream vShaderStream, fShaderStream;
	try
	{
		
		std::ifstream vShaderFile;
		vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		vShaderFile.open(vertexShaderPath);
		vShaderStream << vShaderFile.rdbuf();
		vShaderFile.close();

		std::ifstream fShaderFile;
		fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		fShaderFile.open(fragmentShaderPath);
		fShaderStream << fShaderFile.rdbuf();
		fShaderFile.close();

		vertexCode = vShaderStream.str();
		fragmentCode = fShaderStream.str();

		uniforms = gatherUniforms(vShaderStream);
		uniforms.merge(gatherUniforms(fShaderStream), [](const Uniform& lhs, const Uniform& rhs) { return lhs.name < rhs.name && lhs.type < rhs.type; });
	}
	catch (const std::ifstream::failure& e)
	{
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ, cause: " << e.what() << std::endl;
	}
	const char* vShaderCode = vertexCode.c_str();
	const char* fShaderCode = fragmentCode.c_str();
	const GLuint vertex = compileVertexShader(vShaderCode);
	const GLuint fragment = compileFragmentShader(fShaderCode);
	linkProgram(vertex, fragment);

	queryUniformLocations(uniforms);
}

GLuint Shader::compileVertexShader(const char* vertexShaderSource)
{
	const GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
	glCompileShader(vertexShader);
	GLint success;
	GLchar infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED/n" << infoLog << std::endl;
	}

	return vertexShader;
}

GLuint Shader::compileFragmentShader(const char* fragmentShaderSource)
{
	const GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
	glCompileShader(fragmentShader);

	GLint success;
	GLchar infoLog[512];
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED/n" << infoLog << std::endl;
	}

	return fragmentShader;
}

std::list<Uniform> Shader::gatherUniforms(std::stringstream& stream) const
{
	std::list<Uniform> uniforms;
	for (std::string line; std::getline(stream, line); )
	{
		std::stringstream line2(line);
		std::string word;
		for (; std::getline(line2, word, ' ');)
		{
			if (word == "uniform")
			{
				Uniform uniform;
				std::string type;
				std::getline(line2, uniform.type, ' ');
				std::string uniformName;
				std::getline(line2, uniform.name, ';');
				uniforms.push_back(uniform);
			}
		}
	}
	return uniforms;
}

void Shader::linkProgram(GLuint vertexShader, GLuint fragmentShader)
{
	ID = glCreateProgram();

	glAttachShader(ID, vertexShader);
	glAttachShader(ID, fragmentShader);
	glLinkProgram(ID);

	GLint success;
	GLchar infoLog[512];
	glGetProgramiv(ID, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(ID, 512, NULL, infoLog);
		std::cout << "ERROR::PROGRAM::LINKAGE_FAILED/n" << infoLog << std::endl;
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}

void Shader::queryUniformLocations(const std::list<Uniform>& uniforms)
{
	for (const auto& uniform : uniforms)
	{
		uniformLocations.emplace(uniform, glGetUniformLocation(ID, uniform.name.c_str()));
	}	
}

Shader::~Shader()
{
	glDeleteProgram(ID);
#ifdef DEBUG
	std::cout << "Shader deleted!" << std::endl;
#endif
}

void Shader::use() const
{
	glUseProgram(ID);
}

GLuint Shader::getLocation(const std::string& name) const
{
	const auto uniform_it = std::find_if(std::begin(uniformLocations), std::end(uniformLocations),
		[&name](const std::pair<Uniform, GLuint>& pair)
		{
			return pair.first.name == name;
		});

	if (uniform_it != std::end(uniformLocations))
	{
		return uniform_it->second;
	}
	return 0;
}

void Shader::setBool(const std::string& name, bool value) const
{
	glUniform1i(getLocation(name), value);
}

void Shader::setInt(const std::string& name, int value) const
{
	glUniform1i(getLocation(name), value);
}

void Shader::setFloat(const std::string& name, float value) const
{
	glUniform1f(getLocation(name), value);
}

void Shader::setVec2(const std::string& name, glm::vec2 value) const
{
	glUniform2fv(getLocation(name), 1, glm::value_ptr(value));
}

void Shader::setVec3(const std::string& name, glm::vec3 value) const
{
	glUniform3fv(getLocation(name), 1, glm::value_ptr(value));
}

void Shader::setMat4(const std::string& name, glm::mat4 value) const
{
	glUniformMatrix4fv(getLocation(name), 1, GL_FALSE, glm::value_ptr(value));
}

}