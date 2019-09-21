#ifndef SPARK_RENDERER_H
#define SPARK_RENDERER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Structs.h>
#include <Shader.h>
#include <ModelMesh.h>
#include <Camera.h>
#include <functional>
#include "Enums.h"

class SparkRenderer
{
private:
	ScreenQuad screenQuad{};

	GLuint mainFramebuffer{}, colorTexture{}, positionTexture{}, normalsTexture{};
	GLuint postprocessingFramebuffer{}, postProcessingTexture{};
	
	std::weak_ptr<Shader> mainShader;
	std::weak_ptr<Shader> screenShader;
	std::weak_ptr<Shader> postprocessingShader;

	void createTexture(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping, GLenum textureSampling);

	void postprocessingPass();
	void renderToScreen();

	void initMembers();
	void createFrameBuffersAndTextures();
	void deleteFrameBuffersAndTextures() const;
	~SparkRenderer() = default;
	SparkRenderer() = default;
public:
	static GLFWwindow* window;
	static std::map<ShaderType, std::list<std::function<void(std::shared_ptr<Shader>&)>>> renderQueue;
	
	static SparkRenderer* getInstance();
	static void resizeWindow(GLuint width, GLuint height);
	SparkRenderer(const SparkRenderer&) = delete;
	SparkRenderer operator=(const SparkRenderer&) = delete;
	
	static void initOpenGL();
	void setup();
	void renderPass();
	void cleanup();

	static bool isWindowOpened();

	static void error_callback(int error, const char* description);
};

#endif