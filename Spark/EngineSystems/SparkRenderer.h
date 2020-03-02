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
    void updateBufferBindings() const;
    void drawGui();

    bool ssaoEnable = true;
    int kernelSize = 24;
    float radius = 0.45f;
    float bias = 0.015f;
    float power = 5.0f;

private:
	ScreenQuad screenQuad{};

	GLuint mainFramebuffer{}, colorTexture{}, normalsTexture{}, depthTexture{};
    GLuint lightFrameBuffer{}, lightColorTexture{};
    GLuint postprocessingFramebuffer{}, postProcessingTexture{};
    GLuint motionBlurFramebuffer{}, motionBlurTexture{};
    GLuint lightShaftFramebuffer{}, lightShaftTexture{};
    GLuint gaussianBlurFramebuffer{}, gaussianBlurFramebuffer2{}, horizontalBlurTexture{}, verticalBlurTexture{};
    GLuint ssaoFramebuffer{}, ssaoTexture{}, randomNormalsTexture{};

    GLuint textureHandle{};  // temporary, its only a handle to other texture -> dont delete it

	std::weak_ptr<Shader> mainShader;
	std::weak_ptr<Shader> screenShader;
	std::weak_ptr<Shader> postprocessingShader;
	std::weak_ptr<Shader> lightShader;
	std::weak_ptr<Shader> motionBlurShader;
	std::weak_ptr<Shader> cubemapShader;
    std::weak_ptr<Shader> ssaoShader;
	Cube cube = Cube();
    UniformBuffer uniformBuffer{};
    UniformBuffer sampleUniformBuffer{};

	~SparkRenderer() = default;
	SparkRenderer() = default;

	void createTexture(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping,
                       GLenum textureSampling);
    void fillGBuffer();
    void ssaoComputing() const;
    void renderLights() const;
    void renderCubemap() const;
    void postprocessingPass();
    void lightShafts();
    void motionBlur();
    void renderToScreen() const;
    void initMembers();
    void resizeWindowIfNecessary();
    void blurTexture(GLuint texture) const;
    void createFrameBuffersAndTextures();
    void createFramebuffer(GLuint& framebuffer, const std::vector<GLuint>&& colorTextures, GLuint renderbuffer = 0);
    void bindDepthTexture(GLuint& framebuffer, GLuint depthTexture);
    void deleteFrameBuffersAndTextures() const;
};
}
#endif