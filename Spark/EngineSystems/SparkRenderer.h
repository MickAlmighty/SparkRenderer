#ifndef SPARK_RENDERER_H
#define SPARK_RENDERER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Enums.h"
#include "ModelMesh.h"
#include "Structs.h"
#include "Shader.h"

namespace spark {

class SparkRenderer
{
public:
	std::map<ShaderType, std::list<std::function<void(std::shared_ptr<Shader>&)>>> renderQueue;
	
	SparkRenderer(const SparkRenderer&) = delete;
	SparkRenderer operator=(const SparkRenderer&) = delete;

	void setup();
	void renderPass();
	void cleanup();

	static SparkRenderer* getInstance();

private:
	ScreenQuad screenQuad{};

	GLuint mainFramebuffer{}, colorTexture{}, positionTexture{}, normalsTexture{}, roughnessTexture{}, metalnessTexture{};
	GLuint postprocessingFramebuffer{}, postProcessingTexture{};

	std::weak_ptr<Shader> mainShader;
	std::weak_ptr<Shader> screenShader;
	std::weak_ptr<Shader> postprocessingShader;

	~SparkRenderer() = default;
	SparkRenderer() = default;

	void createTexture(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping, GLenum textureSampling);
	void postprocessingPass();
	void renderToScreen();
	void initMembers();
	void createFrameBuffersAndTextures();
	void deleteFrameBuffersAndTextures() const;
};
}
#endif