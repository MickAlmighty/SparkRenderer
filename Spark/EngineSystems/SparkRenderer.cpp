#include "EngineSystems/SparkRenderer.h"

#include <iostream>
#include <exception>
#include <functional>

#include <GUI/ImGuizmo.h>
#include <GUI/ImGui/imgui.h>
#include <GUI/ImGui/imgui_impl_glfw.h>
#include <GUI/ImGui/imgui_impl_opengl3.h>

#include "Camera.h"
#include "EngineSystems/ResourceManager.h"
#include "EngineSystems/SceneManager.h"
#include "HID.h"
#include "Scene.h"
#include "Spark.h"
#include "Clock.h"

namespace spark {

SparkRenderer* SparkRenderer::getInstance()
{
	static SparkRenderer* spark_renderer = nullptr;
	if (spark_renderer == nullptr)
	{
		spark_renderer = new SparkRenderer();
	}
	return  spark_renderer;
}

void SparkRenderer::setup()
{
	initMembers();
}

void SparkRenderer::createTexture(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format,
	GLenum pixelFormat, GLenum textureWrapping, GLenum textureSampling)
{
	glCreateTextures(GL_TEXTURE_2D, 1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, pixelFormat, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, textureSampling);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, textureSampling);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, textureWrapping);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, textureWrapping);
}

void SparkRenderer::initMembers()
{
	screenQuad.setup();
	mainShader = ResourceManager::getInstance()->getShader(ShaderType::DEFAULT_SHADER);
	screenShader = ResourceManager::getInstance()->getShader(ShaderType::SCREEN_SHADER);
	postprocessingShader = ResourceManager::getInstance()->getShader(ShaderType::POSTPROCESSING_SHADER);
	lightShader = ResourceManager::getInstance()->getShader(ShaderType::LIGHT_SHADER);
	motionBlurShader = ResourceManager::getInstance()->getShader(ShaderType::MOTION_BLUR_SHADER);
	cubemapShader = ResourceManager::getInstance()->getShader(ShaderType::CUBEMAP_SHADER);
	createFrameBuffersAndTextures();
}

void SparkRenderer::createFrameBuffersAndTextures()
{
	glCreateFramebuffers(1, &mainFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, mainFramebuffer);
	createTexture(colorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
	createTexture(positionTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
	createTexture(normalsTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
	createTexture(roughnessTexture, Spark::WIDTH, Spark::HEIGHT, GL_R16F, GL_RED, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
	createTexture(metalnessTexture, Spark::WIDTH, Spark::HEIGHT, GL_R16F, GL_RED, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, positionTexture, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, colorTexture, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, normalsTexture, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, roughnessTexture, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, metalnessTexture, 0);

	GLuint renderbuffer;
	glCreateRenderbuffers(1, &renderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, Spark::WIDTH, Spark::HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderbuffer);

	GLenum attachments[6] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4 };
	glDrawBuffers(6, attachments);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		throw std::exception("Main framebuffer incomplete!");
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glCreateFramebuffers(1, &lightFrameBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, lightFrameBuffer);
	createTexture(lightColorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, lightColorTexture, 0);
	GLenum attachments2[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, attachments2);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		throw std::exception("Postprocessing framebuffer incomplete!");
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	glCreateFramebuffers(1, &postprocessingFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, postprocessingFramebuffer);
	createTexture(postProcessingTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, postProcessingTexture, 0);
	GLenum attachments3[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, attachments3);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		throw std::exception("Postprocessing framebuffer incomplete!");
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glCreateFramebuffers(1, &cubemapFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, cubemapFramebuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, lightColorTexture, 0);
	GLenum attachments4[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, attachments4);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		throw std::exception("Cubemap framebuffer incomplete!");
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glCreateFramebuffers(1, &motionBlurFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, motionBlurFramebuffer);
	createTexture(motionBlurTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, motionBlurTexture, 0);
	GLenum attachments5[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, attachments5);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		throw std::exception("Motion Blur framebuffer incomplete!");
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SparkRenderer::deleteFrameBuffersAndTextures() const
{
	GLuint textures[8] = { colorTexture, positionTexture, normalsTexture, roughnessTexture, metalnessTexture, lightColorTexture, postProcessingTexture, motionBlurTexture };
	glDeleteTextures(8, textures);

	GLuint frameBuffers[5] = { mainFramebuffer, lightFrameBuffer, postprocessingFramebuffer, motionBlurFramebuffer, cubemapFramebuffer };
	glDeleteFramebuffers(5, frameBuffers);
}

void SparkRenderer::renderPass()
{
	int width, height;
	glfwGetWindowSize(Spark::window, &width, &height);
	if (Spark::WIDTH != width || Spark::HEIGHT != height)
	{
		Spark::WIDTH = width;
		Spark::HEIGHT = height;
		deleteFrameBuffersAndTextures();
		createFrameBuffersAndTextures();
	}

	glViewport(0, 0, Spark::WIDTH, Spark::HEIGHT);

	const auto camera = SceneManager::getInstance()->getCurrentScene()->getCamera();

	

	const glm::mat4 view = camera->getViewMatrix();
	const glm::mat4 projection = camera->getProjectionMatrix();
	static glm::mat4 prevProjectionView = projection * view;

	glBindFramebuffer(GL_FRAMEBUFFER, mainFramebuffer);
	glClearColor(0, 0, 0, 1);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	std::shared_ptr<Shader> shader = mainShader.lock();
	shader->use();
	shader->setMat4("view", view);
	shader->setMat4("projection", projection);
	for (auto& drawMesh : renderQueue[ShaderType::DEFAULT_SHADER])
	{
		drawMesh(shader);
	}
	renderQueue[ShaderType::DEFAULT_SHADER].clear();

	glDepthMask(GL_FALSE);
	glBindFramebuffer(GL_FRAMEBUFFER, cubemapFramebuffer);
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	const auto cubemapShaderPtr = cubemapShader.lock();
	cubemapShaderPtr->use();
	cubemapShaderPtr->setMat4("view", view);
	cubemapShaderPtr->setMat4("projection", projection);
	const auto cubemap = SceneManager::getInstance()->getCurrentScene()->cubemap;
	if(cubemap)
	{
		glBindTextureUnit(0, cubemap->cubemap);
	}
	else
	{
		glBindTextureUnit(0, 0);
	}
	cube.draw();
	glBindTextures(0, 1, nullptr);
	glDepthMask(GL_TRUE);

	std::string light = "light";
	glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, static_cast<GLsizei>(light.size()), light.c_str());
	glBindFramebuffer(GL_FRAMEBUFFER, lightFrameBuffer);
	const std::shared_ptr<Shader> lShader = lightShader.lock();
	lShader->use();
	lShader->bindSSBO("DirLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->dirLightSSBO);
	lShader->bindSSBO("PointLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->pointLightSSBO);
	lShader->bindSSBO("SpotLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->spotLightSSBO);
	lShader->setVec3("camPos", camera->getPosition());
	if(cubemap)
	{
		std::array<GLuint, 8> textures{ positionTexture, colorTexture, normalsTexture,
		roughnessTexture, metalnessTexture, cubemap->irradianceCubemap,
		cubemap->prefilteredCubemap,
		cubemap->brdfLUTTexture };
		glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
		screenQuad.draw();
		glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);
	}
	else
	{
		std::array<GLuint, 8> textures{ positionTexture, colorTexture, normalsTexture,
		roughnessTexture, metalnessTexture, 0, 0, 0};
		glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
		screenQuad.draw();
		glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);
	}
	glPopDebugGroup();
	postprocessingPass();

	{
		std::string message = "motion blur";
		glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, static_cast<GLsizei>(message.size()), message.c_str());
		const std::shared_ptr<Shader> motionBlurShaderS = motionBlurShader.lock();
		glBindFramebuffer(GL_FRAMEBUFFER, motionBlurFramebuffer);
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		motionBlurShaderS->use();
		motionBlurShaderS->setMat4("viewProjectionMatrix", projection * view);
		motionBlurShaderS->setMat4("previousViewProjectionMatrix", prevProjectionView);
		motionBlurShaderS->setFloat("currentFPS", static_cast<float>(Clock::getFPS()));
		std::array<GLuint, 2> textures2{ postProcessingTexture, positionTexture };
		glBindTextures(0, static_cast<GLsizei>(textures2.size()), textures2.data());
		screenQuad.draw();
		glBindTextures(0, static_cast<GLsizei>(textures2.size()), nullptr);
		glPopDebugGroup();
		prevProjectionView = projection * view;
	}
	renderToScreen();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	glfwSwapBuffers(Spark::window);
}

void SparkRenderer::postprocessingPass()
{
	glBindFramebuffer(GL_FRAMEBUFFER, postprocessingFramebuffer);
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	postprocessingShader.lock()->use();
	postprocessingShader.lock()->setVec2("inversedScreenSize", { 1.0f / Spark::WIDTH, 1.0f / Spark::HEIGHT });

	glBindTextureUnit(0, lightColorTexture);

	screenQuad.draw();
}


void SparkRenderer::renderToScreen()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glClearColor(0, 0, 0, 0);
	//glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	screenShader.lock()->use();

	glBindTextureUnit(0, motionBlurTexture);

	screenQuad.draw();
}

void SparkRenderer::cleanup()
{
	deleteFrameBuffersAndTextures();
}

}
