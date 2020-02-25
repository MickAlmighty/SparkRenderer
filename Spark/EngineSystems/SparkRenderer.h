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
	void cleanup() const;

	static SparkRenderer* getInstance();
    void updateBufferBindings() const;

private:
	ScreenQuad screenQuad{};

	GLuint mainFramebuffer{}, colorTexture{}, positionTexture{}, normalsTexture{}, roughnessTexture{}, metalnessTexture{};
	GLuint lightFrameBuffer{}, lightColorTexture{};
	GLuint postprocessingFramebuffer{}, postProcessingTexture{};
	GLuint motionBlurFramebuffer{}, motionBlurTexture{};
	GLuint textureHandle{}; //temporary its only a handle to other texture -> dont delete it
	GLuint cubemapFramebuffer{};

	std::weak_ptr<Shader> mainShader;
	std::weak_ptr<Shader> screenShader;
	std::weak_ptr<Shader> postprocessingShader;
	std::weak_ptr<Shader> lightShader;
	std::weak_ptr<Shader> motionBlurShader;
	std::weak_ptr<Shader> cubemapShader;
	Cube cube = Cube();

	~SparkRenderer() = default;
	SparkRenderer() = default;

	void createTexture(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping, GLenum textureSampling);
	void renderCubemap() const;
	void renderLights() const;
	void postprocessingPass();
	void motionBlur();
	void renderToScreen() const;
	void initMembers();
	void resizeWindowIfNecessary();
	void createFrameBuffersAndTextures();
	void deleteFrameBuffersAndTextures() const;
};
}
#endif