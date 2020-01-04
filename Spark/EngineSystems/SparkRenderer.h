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
	std::map<ShaderType, std::vector<std::shared_ptr<ModelMesh>>> renderInstancedQueue;
	
	SparkRenderer(const SparkRenderer&) = delete;
	SparkRenderer(const SparkRenderer&&) = delete;
	SparkRenderer operator=(const SparkRenderer&) = delete;
	SparkRenderer operator=(const SparkRenderer&&) = delete;

	void setup();
	void renderPass();
	void cleanup() const;

	static SparkRenderer* getInstance();
	void addMeshDataToBuffer(const std::vector<glm::vec3>& vertices, const std::vector<GLuint>& indices);

private:
	ScreenQuad screenQuad{};

	GLuint mainFramebuffer{}, colorTexture{}, positionTexture{}, normalsTexture{}, roughnessTexture{}, metalnessTexture{};
	GLuint lightFrameBuffer{}, lightColorTexture{};
	GLuint postprocessingFramebuffer{}, postProcessingTexture{};
	GLuint motionBlurFramebuffer{}, motionBlurTexture{};
	GLuint cubemapFramebuffer{};
	GLuint brightPassFramebuffer{}, brightPassTexture{};
	GLuint brightPassHalfFramebuffer{}, brightHalf{};
	GLuint brightPassQuarterFramebuffer{}, brightQuarter{};
	GLuint brightPassOneEightsFramebuffer{}, brightOneEights{};
	GLuint gaussianBlurOneEightsFramebuffer{}, gaussianBlurOneEightsTexture{};
	GLuint gaussianBlurOneEightsFramebuffer2{}, gaussianBlurOneEightsTexture2{};
	GLuint gaussianBlurQuarterFramebuffer{}, gaussianBlurQuarterTexture{};
	GLuint gaussianBlurQuarterFramebuffer2{}, gaussianBlurQuarterTexture2{};
	GLuint gaussianBlurHalfFramebuffer{}, gaussianBlurHalfTexture{};
	GLuint gaussianBlurHalfFramebuffer2{}, gaussianBlurHalfTexture2{};
	GLuint textureHandle{}; //temporary its only a handle to other texture -> dont delete it
	SSBO instancedSSBO{};

	std::weak_ptr<Shader> mainShader;
	std::weak_ptr<Shader> screenShader;
	std::weak_ptr<Shader> postprocessingShader;
	std::weak_ptr<Shader> lightShader;
	std::weak_ptr<Shader> motionBlurShader;
	std::weak_ptr<Shader> cubemapShader;
	std::weak_ptr<Shader> defaultInstancedShader;
	Cube cube = Cube();
	MultiDrawIndirectBuffer bufer{};
	MultiDrawInstancedIndirectBuffer instancedIndirectBuffer{};

	~SparkRenderer() = default;
	SparkRenderer() = default;

	void createTexture(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping, GLenum textureSampling);
	void renderLights() const;
	void bloom() const;
	void renderCubemap() const;
	void postprocessingPass();
	void motionBlur();
	void renderToScreen() const;
	void initMembers();
	void resizeWindowIfNecessary();
	void createFrameBuffersAndTextures();
	void createFramebuffer(GLuint& framebuffer, const std::vector<GLuint>&& colorTextures, GLuint renderbuffer = 0);
	void deleteFrameBuffersAndTextures() const;
};
}
#endif