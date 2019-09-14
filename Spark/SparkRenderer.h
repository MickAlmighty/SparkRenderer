#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Structs.h"
#include "Shader.h"
#include "ModelMesh.h"
#include "Camera.h"

class SparkRenderer
{
private:
	ScreenQuad screenQuad{};
	std::shared_ptr<ModelMesh> model = nullptr;

	GLuint mainFramebuffer{}, colorTexture{}, positionTexture{}, normalsTexture{};
	GLuint postprocessingFramebuffer{}, postProcessingTexture{};
	
	
	
	std::shared_ptr<Shader> mainShader = nullptr;
	std::shared_ptr<Shader> screenShader = nullptr;
	std::shared_ptr<Shader> postprocessingShader = nullptr;

	void createTexture(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping, GLenum textureSampling);

	void postprocessingPass();
	void renderToScreen();

	void initMembers();
	~SparkRenderer() = default;
	SparkRenderer() = default;
public:
	static GLFWwindow* window;
	static SparkRenderer* getInstance();
	SparkRenderer(const SparkRenderer&) = delete;
	SparkRenderer operator=(const SparkRenderer&) = delete;
	Camera* camera = nullptr;
	static void initOpenGL();

	void setup();
	void renderPass();
	void cleanup();

	static bool isWindowOpened();

	static void error_callback(int error, const char* description);
};

