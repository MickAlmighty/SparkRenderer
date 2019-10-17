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
	createFrameBuffersAndTextures();
}

void SparkRenderer::createFrameBuffersAndTextures()
{
	glCreateFramebuffers(1, &mainFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, mainFramebuffer);
	createTexture(colorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
	createTexture(positionTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
	createTexture(normalsTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
	createTexture(roughnessTexture, Spark::WIDTH, Spark::HEIGHT, GL_R16F, GL_RED, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
	createTexture(metalnessTexture, Spark::WIDTH, Spark::HEIGHT, GL_R16F, GL_RED, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
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

	GLenum attachments[5] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4 };
	glDrawBuffers(5, attachments);

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
}

void SparkRenderer::deleteFrameBuffersAndTextures() const
{
	GLuint textures[7] = { colorTexture, positionTexture, normalsTexture, roughnessTexture, metalnessTexture,lightColorTexture, postProcessingTexture };
	glDeleteTextures(7, textures);

	GLuint frameBuffers[3] = { mainFramebuffer, lightFrameBuffer, postprocessingFramebuffer };
	glDeleteFramebuffers(3, frameBuffers);
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

	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGuizmo::BeginFrame();

	const auto camera = SceneManager::getInstance()->getCurrentScene()->getCamera();

	glBindFramebuffer(GL_FRAMEBUFFER, mainFramebuffer);
	glClearColor(0, 0, 0, 1);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	const glm::mat4 view = camera->getViewMatrix();
	const glm::mat4 projection = camera->getProjectionMatrix();


	std::shared_ptr<Shader> shader = mainShader.lock();
	shader->use();
	shader->setMat4("view", view);
	shader->setMat4("projection", projection);
	for (auto& drawMesh : renderQueue[ShaderType::DEFAULT_SHADER])
	{
		drawMesh(shader);
	}
	renderQueue[ShaderType::DEFAULT_SHADER].clear();

	std::string light = "light";
	glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, light.size(), light.c_str());
	glBindFramebuffer(GL_FRAMEBUFFER, lightFrameBuffer);
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	const std::shared_ptr<Shader> lShader = lightShader.lock();
	lShader->use();
	lShader->setVec3("camPos", camera->getPosition());
	GLuint textures[5] = {positionTexture, colorTexture, normalsTexture, roughnessTexture, metalnessTexture};
	glBindTextures(0, 5, textures);
	screenQuad.draw();
	glBindTextures(0, 5, nullptr);

	glPopDebugGroup();
	postprocessingPass();
	renderToScreen();

	SceneManager::getInstance()->drawGUI();

	//if(ImGui::Begin("EditorWindow", 0, ImGuiWindowFlags_NoScrollbar))
	//{
	//	auto size = ImGui::GetWindowSize();
	//	//size.x -= 35;
	//	size.y -= 40;
	//	ImGui::Image((void*)(intptr_t)postProcessingTexture, ImVec2(size.y / (9.0f / 16.0f) - 30, size.y), ImVec2(0,1), ImVec2(1,0));

	//	
	//}
	//ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	glfwSwapBuffers(Spark::window);
}

void SparkRenderer::postprocessingPass()
{
	glBindFramebuffer(GL_FRAMEBUFFER, postprocessingFramebuffer);
	glClearColor(0, 0, 0, 1);
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
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	screenShader.lock()->use();

	glBindTextureUnit(0, postProcessingTexture);

	screenQuad.draw();
}

void SparkRenderer::cleanup()
{
	deleteFrameBuffersAndTextures();
}

}