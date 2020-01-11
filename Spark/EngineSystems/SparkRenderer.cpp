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

	inline static int debug_group_counter = 0;

#define PUSH_DEBUG_GROUP(x) { \
	std::string message = #x; \
	glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, debug_group_counter, static_cast<GLsizei>(message.length()), message.data()); \
	debug_group_counter++; }

#define POP_DEBUG_GROUP() \
	glPopDebugGroup(); \
	debug_group_counter--;

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
		PROFILE_FUNCTION();
		initMembers();
	}

	void SparkRenderer::initMembers()
	{
		PROFILE_FUNCTION();
		screenQuad.setup();
		mainShader = ResourceManager::getInstance()->getShader(ShaderType::DEFAULT_SHADER);
		screenShader = ResourceManager::getInstance()->getShader(ShaderType::SCREEN_SHADER);
		postprocessingShader = ResourceManager::getInstance()->getShader(ShaderType::POSTPROCESSING_SHADER);
		lightShader = ResourceManager::getInstance()->getShader(ShaderType::LIGHT_SHADER);
		motionBlurShader = ResourceManager::getInstance()->getShader(ShaderType::MOTION_BLUR_SHADER);
		cubemapShader = ResourceManager::getInstance()->getShader(ShaderType::CUBEMAP_SHADER);
		defaultInstancedShader = ResourceManager::getInstance()->getShader(ShaderType::DEFAULT_INSTANCED_SHADER);

		const std::shared_ptr<Shader> lShader = lightShader.lock();
		lShader->use();
		lShader->bindSSBO("DirLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->dirLightSSBO.ID);
		lShader->bindSSBO("PointLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->pointLightSSBO.ID);
		lShader->bindSSBO("SpotLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->spotLightSSBO.ID);

		const std::shared_ptr<Shader> instancedShader = defaultInstancedShader.lock();
		instancedShader->use();
		instancedShader->bindSSBO("Models", instancedSSBO.ID);

		createFrameBuffersAndTextures();

		sortingMeshes = std::thread([this]()
		{
			int counter = 0;
			while(rendering)
			{
				while (!renderInstancedQueue.empty())
				{
					auto modelMeshPtr = this->popMeshFromInstancedQueue();

					for (const auto& mesh : modelMeshPtr->meshes)
					{
						if (instancedMeshes.find(mesh) != instancedMeshes.end())
						{
							instancedMeshes[mesh] += 1;
						}
						else
						{
							instancedMeshes.insert(std::make_pair(mesh, 1));
						}
						models.push_back(modelMeshPtr->getGameObject()->transform.world.getMatrix());
					}
					
					++counter;
					
					if (this->renderPassStarted && !renderInstancedQueue.empty())
					{
						//printf("sorting!\n");
						sortingDone = false;
						cv.notify_all();
					}
				}

				if (this->renderPassStarted && renderInstancedQueue.empty())
				{
					//printf("Sorting done! Sorted %d modelMeshes.\n", counter);
					counter = 0;
					sortingDone = true;
					cv.notify_all();

					std::unique_lock<std::mutex> lk(cv_m);
					cv.wait(lk, [this]
					{
						//printf("Waiting for end of render pass\n");
						return !renderPassStarted.load();
					});
					models.clear();
					instancedMeshes.clear();
				}
			}
		});
	}

	void SparkRenderer::renderPass()
	{
		PROFILE_FUNCTION();
		renderPassStarted = true;
		resizeWindowIfNecessary();

		const auto camera = SceneManager::getInstance()->getCurrentScene()->getCamera();

		const glm::mat4 view = camera->getViewMatrix();
		const glm::mat4 projection = camera->getProjectionMatrix();

		PUSH_DEBUG_GROUP(RENDER_TO_MAIN_FRAMEBUFFER);
		glBindFramebuffer(GL_FRAMEBUFFER, mainFramebuffer);
		glClearColor(0, 0, 0, 1);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		{
			PROFILE_SCOPE("Normal rendering");
			std::shared_ptr<Shader> shader = mainShader.lock();
			shader->use();
			shader->setMat4("view", view);
			shader->setMat4("projection", projection);
			for (auto& drawMesh : renderQueue[ShaderType::DEFAULT_SHADER])
			{
				drawMesh(shader);
			}
			renderQueue[ShaderType::DEFAULT_SHADER].clear();
			POP_DEBUG_GROUP();
		}

		{
			PROFILE_SCOPE("Waiting for sorted meshes");
			std::unique_lock<std::mutex> lk(cv_m);
			cv.wait(lk, [this]
			{
				//printf("waiting on sorting done\n");
				return sortingDone.load();
			});
		}
		if (!models.empty())
		{
			PROFILE_SCOPE("Instanced rendering");
			const std::shared_ptr<Shader> instancedShader = defaultInstancedShader.lock();
			instancedShader->use();
			instancedShader->setMat4("view", view);
			instancedShader->setMat4("projection", projection);
			
			instancedSSBO.update(models);
			for (const auto& [mesh, instances] : instancedMeshes)
			{
				mesh.bindTextures();
				instancedIndirectBuffer.addMesh(mesh);
				instancedIndirectBuffer.drawInstances(mesh, instances);
				instancedIndirectBuffer.cleanup();
			}
		}
		glDisable(GL_CULL_FACE);
		renderLights();
		renderCubemap();
		//bloom();

		PUSH_DEBUG_GROUP(RENDER_PATHS);
		glBindFramebuffer(GL_FRAMEBUFFER, lightFrameBuffer);

		const std::shared_ptr<Shader> pathShader = ResourceManager::getInstance()->getShader(ShaderType::PATH_SHADER);
		pathShader->use();
		pathShader->setMat4("VP", projection * view);
		/*for (auto& drawPath : renderQueue[ShaderType::PATH_SHADER])
		{
			drawPath(shader);
		}
		renderQueue[ShaderType::PATH_SHADER].clear();*/
		bufer.draw();
		bufer.clear();
		bufer = MultiDrawIndirectBuffer();
		POP_DEBUG_GROUP();

		postprocessingPass();
		motionBlur();
		renderToScreen();

		{
			PROFILE_SCOPE("RenderPass::DrawGUI");
			if (Spark::gui)
			{
				PUSH_DEBUG_GROUP(GUI);
				ImGui::Render();
				ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
				POP_DEBUG_GROUP();
			}
		}

		PROFILE_SCOPE("Swap Buffers");
		glfwSwapBuffers(Spark::window);
		renderPassStarted = false;
		cv.notify_all();
	}

	void SparkRenderer::resizeWindowIfNecessary()
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
	}

	void SparkRenderer::renderLights() const
	{
		PROFILE_FUNCTION();

		PUSH_DEBUG_GROUP(PBR_LIGHT);

		glBindFramebuffer(GL_FRAMEBUFFER, lightFrameBuffer);
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT);

		const auto camera = SceneManager::getInstance()->getCurrentScene()->getCamera();
		const auto cubemap = SceneManager::getInstance()->getCurrentScene()->cubemap;

		glDisable(GL_DEPTH_TEST);
		const std::shared_ptr<Shader> lShader = lightShader.lock();
		lShader->use();
		lShader->setVec3("camPos", camera->getPosition());
		if (cubemap)
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
			roughnessTexture, metalnessTexture, 0, 0, 0 };
			glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
			screenQuad.draw();
			glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);
		}
		glEnable(GL_DEPTH_TEST);

		POP_DEBUG_GROUP();
	}

	void SparkRenderer::renderCubemap() const
	{
		PROFILE_FUNCTION();
		const auto cubemap = SceneManager::getInstance()->getCurrentScene()->cubemap;
		if (!cubemap)
			return;

		const auto camera = SceneManager::getInstance()->getCurrentScene()->getCamera();
		const glm::mat4 view = camera->getViewMatrix();
		const glm::mat4 projection = camera->getProjectionMatrix();
		const glm::vec3 cameraPosition = camera->getPosition();
		const glm::vec3 cameraFront = camera->getFront();
		const float farPlane = camera->getFarPlane();
		PUSH_DEBUG_GROUP(RENDER_CUBEMAP);
		glBindFramebuffer(GL_FRAMEBUFFER, cubemapFramebuffer);

		glDepthFunc(GL_LEQUAL);
		const auto cubemapShaderPtr = cubemapShader.lock();
		cubemapShaderPtr->use();
		cubemapShaderPtr->setMat4("view", view);
		cubemapShaderPtr->setMat4("projection", projection);
		cubemapShaderPtr->setVec3("cubemapPosition", cameraPosition + cameraFront * farPlane);

		glBindTextureUnit(0, cubemap->cubemap);
		cube.draw();
		glBindTextures(0, 1, nullptr);
		glDepthFunc(GL_LESS);

		POP_DEBUG_GROUP();
	}

	void SparkRenderer::bloom() const
	{
		PROFILE_FUNCTION();
		PUSH_DEBUG_GROUP(BRIGHT_PASS);
		glBindFramebuffer(GL_FRAMEBUFFER, brightPassFramebuffer);
		//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brightPassFramebuffer, 0);
		glClearColor(0.0, 0.0, 0.0, 0.0);
		glClear(GL_COLOR_BUFFER_BIT);

		const auto brightPassShader = ResourceManager::getInstance()->getShader(ShaderType::BRIGHT_PASS_SHADER);
		brightPassShader->use();

		glBindTextureUnit(0, lightColorTexture);
		screenQuad.draw();
		glBindTextures(0, 1, nullptr);

		{
			PUSH_DEBUG_GROUP(DOWN_SCALE);
			const auto downScaleShader = ResourceManager::getInstance()->getShader(ShaderType::DOWNSCALE_SHADER);
			downScaleShader->use();
			downScaleShader->setBool("blend", false);
			glBindFramebuffer(GL_FRAMEBUFFER, brightPassOneEightsFramebuffer);
			glViewport(0, 0, Spark::WIDTH / 8, Spark::HEIGHT / 8);
			glBindTextureUnit(0, brightPassTexture);
			screenQuad.draw();
			glBindTextures(0, 1, nullptr);

				PUSH_DEBUG_GROUP(GAUSSIAN_BLUR_1_8);
				glBindFramebuffer(GL_FRAMEBUFFER, gaussianBlurOneEightsFramebuffer);
				const auto gaussianBlurShader = ResourceManager::getInstance()->getShader(ShaderType::GAUSSIAN_BLUR_SHADER);
				gaussianBlurShader->use();
				gaussianBlurShader->setVec2("inverseScreenSize", { 1.0f / (Spark::WIDTH / 8), 1.0f / (Spark::HEIGHT / 8) });
				gaussianBlurShader->setVec2("direction", { 1.0f, 0.0f });
				glBindTextureUnit(0, brightOneEights);
				screenQuad.draw();
				glBindTextures(0, 1, nullptr);

				glBindFramebuffer(GL_FRAMEBUFFER, gaussianBlurOneEightsFramebuffer2);
				gaussianBlurShader->setVec2("direction", { 0.0f, 1.0f });
				glBindTextureUnit(0, gaussianBlurOneEightsTexture);
				screenQuad.draw();
				glBindTextures(0, 1, nullptr);
				POP_DEBUG_GROUP();

			glBindFramebuffer(GL_FRAMEBUFFER, brightPassQuarterFramebuffer);
			glViewport(0, 0, Spark::WIDTH / 4, Spark::HEIGHT / 4);
			downScaleShader->use();
			downScaleShader->setBool("blend", true);
			glBindTextureUnit(0, brightPassTexture);
			glBindTextureUnit(1, gaussianBlurOneEightsTexture2);
			screenQuad.draw();

				PUSH_DEBUG_GROUP(GAUSSIAN_BLUR_1_4);
				gaussianBlurShader->use();
				gaussianBlurShader->setVec2("inverseScreenSize", { 1.0f / (Spark::WIDTH / 4), 1.0f / (Spark::HEIGHT / 4) });
				gaussianBlurShader->setVec2("direction", { 1.0f, 0.0f });
				glBindFramebuffer(GL_FRAMEBUFFER, gaussianBlurQuarterFramebuffer);
				glBindTextureUnit(0, brightQuarter);
				screenQuad.draw();
				glBindTextures(0, 1, nullptr);

				glBindFramebuffer(GL_FRAMEBUFFER, gaussianBlurQuarterFramebuffer2);
				gaussianBlurShader->setVec2("direction", { 0.0f, 1.0f });
				glBindTextureUnit(0, gaussianBlurQuarterTexture);
				screenQuad.draw();
				glBindTextures(0, 1, nullptr);
				POP_DEBUG_GROUP();

			glBindFramebuffer(GL_FRAMEBUFFER, brightPassHalfFramebuffer);
			glViewport(0, 0, Spark::WIDTH / 2, Spark::HEIGHT / 2);
			downScaleShader->use();
			glBindTextureUnit(0, brightPassTexture);
			glBindTextureUnit(1, gaussianBlurQuarterTexture2);
			screenQuad.draw();

				PUSH_DEBUG_GROUP(GAUSSIAN_BLUR_1_2);
				glBindFramebuffer(GL_FRAMEBUFFER, gaussianBlurHalfFramebuffer);
				gaussianBlurShader->use();
				gaussianBlurShader->setVec2("inverseScreenSize", { 1.0f / (Spark::WIDTH / 2), 1.0f / (Spark::HEIGHT / 2) });
				gaussianBlurShader->setVec2("direction", { 1.0f, 0.0f });
				glBindTextureUnit(0, brightHalf);
				screenQuad.draw();
				glBindTextures(0, 1, nullptr);

				glBindFramebuffer(GL_FRAMEBUFFER, gaussianBlurHalfFramebuffer2);
				gaussianBlurShader->setVec2("direction", { 0.0f, 1.0f });
				glBindTextureUnit(0, gaussianBlurHalfTexture);
				screenQuad.draw();
				glBindTextures(0, 1, nullptr);
				POP_DEBUG_GROUP();

			POP_DEBUG_GROUP();
		}
		POP_DEBUG_GROUP();

		glViewport(0, 0, Spark::WIDTH, Spark::HEIGHT);
	}

	void SparkRenderer::postprocessingPass()
	{
		PROFILE_FUNCTION();
		PUSH_DEBUG_GROUP(POSTPROCESSING);
		/*glBindFramebuffer(GL_FRAMEBUFFER, lightFrameBuffer);
		const auto downScaleShader = ResourceManager::getInstance()->getShader(ShaderType::DOWNSCALE_SHADER);
		downScaleShader->use();
		downScaleShader->setBool("blend", true);
		glBindTextureUnit(0, lightColorTexture);
		glBindTextureUnit(1, gaussianBlurHalfTexture2);
		screenQuad.draw();
		glBindTextures(0, 2, nullptr);*/

		textureHandle = postProcessingTexture;

		glBindFramebuffer(GL_FRAMEBUFFER, postprocessingFramebuffer);
		glDisable(GL_DEPTH_TEST);

		postprocessingShader.lock()->use();
		postprocessingShader.lock()->setVec2("inversedScreenSize", { 1.0f / Spark::WIDTH, 1.0f / Spark::HEIGHT });

		glBindTextureUnit(0, lightColorTexture);
		glBindTextureUnit(1, gaussianBlurHalfTexture2);

		screenQuad.draw();

		POP_DEBUG_GROUP();
	}

	void SparkRenderer::motionBlur()
	{
		PROFILE_FUNCTION();

		const auto camera = SceneManager::getInstance()->getCurrentScene()->getCamera();
		const glm::mat4 projectionView = camera->getProjectionMatrix() * camera->getViewMatrix();
		static glm::mat4 prevProjectionView = projectionView;
		for(unsigned int counter = 0, i = 0; i < 4; ++i)
		{
			glm::vec4 currentColumn = projectionView[i];
			glm::vec4 previousColumn = prevProjectionView[i];
			
			if (currentColumn == previousColumn)
			{
				++counter;
			}
			if (counter == 4)
			{
				return;
			}
		}

		textureHandle = motionBlurTexture;

		PUSH_DEBUG_GROUP(MOTION_BLUR);
		const std::shared_ptr<Shader> motionBlurShaderS = motionBlurShader.lock();
		glBindFramebuffer(GL_FRAMEBUFFER, motionBlurFramebuffer);
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		motionBlurShaderS->use();
		motionBlurShaderS->setMat4("viewProjectionMatrix", projectionView);
		motionBlurShaderS->setMat4("previousViewProjectionMatrix", prevProjectionView);
		motionBlurShaderS->setFloat("currentFPS", static_cast<float>(Clock::getFPS()));
		std::array<GLuint, 2> textures2{ postProcessingTexture, positionTexture };
		glBindTextures(0, static_cast<GLsizei>(textures2.size()), textures2.data());
		screenQuad.draw();
		glBindTextures(0, static_cast<GLsizei>(textures2.size()), nullptr);
		
		prevProjectionView = projectionView;
		POP_DEBUG_GROUP();
	}

	void SparkRenderer::renderToScreen() const
	{
		PROFILE_FUNCTION();
		PUSH_DEBUG_GROUP(RENDER_TO_SCREEN);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDisable(GL_DEPTH_TEST);

		screenShader.lock()->use();

		glBindTextureUnit(0, textureHandle);

		screenQuad.draw();
		POP_DEBUG_GROUP();
	}

	void SparkRenderer::createFrameBuffersAndTextures()
	{
		PROFILE_FUNCTION();
		GLuint renderBuffer{0};
		glCreateRenderbuffers(1, &renderBuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, renderBuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, Spark::WIDTH, Spark::HEIGHT);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBuffer);

		createTexture(colorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
		createTexture(positionTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
		createTexture(normalsTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
		createTexture(roughnessTexture, Spark::WIDTH, Spark::HEIGHT, GL_R16F, GL_RED, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
		createTexture(metalnessTexture, Spark::WIDTH, Spark::HEIGHT, GL_R16F, GL_RED, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);

		createTexture(lightColorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
		createTexture(postProcessingTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
		createTexture(motionBlurTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
		createTexture(brightPassTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
		createTexture(brightHalf, Spark::WIDTH / 2, Spark::HEIGHT / 2, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
		createTexture(brightQuarter, Spark::WIDTH / 4, Spark::HEIGHT / 4, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
		createTexture(brightOneEights, Spark::WIDTH / 8, Spark::HEIGHT / 8, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
		createTexture(gaussianBlurHalfTexture, Spark::WIDTH / 2, Spark::HEIGHT / 2, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
		createTexture(gaussianBlurQuarterTexture, Spark::WIDTH / 4, Spark::HEIGHT / 4, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
		createTexture(gaussianBlurOneEightsTexture, Spark::WIDTH / 8, Spark::HEIGHT / 8, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

		createTexture(gaussianBlurHalfTexture2, Spark::WIDTH / 2, Spark::HEIGHT / 2, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
		createTexture(gaussianBlurQuarterTexture2, Spark::WIDTH / 4, Spark::HEIGHT / 4, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
		createTexture(gaussianBlurOneEightsTexture2, Spark::WIDTH / 8, Spark::HEIGHT / 8, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

		createFramebuffer(mainFramebuffer, { positionTexture, colorTexture, normalsTexture, roughnessTexture, metalnessTexture }, renderBuffer);
		createFramebuffer(lightFrameBuffer, { lightColorTexture }, renderBuffer);
		createFramebuffer(postprocessingFramebuffer, { postProcessingTexture });
		createFramebuffer(cubemapFramebuffer, { lightColorTexture, positionTexture }, renderBuffer);
		createFramebuffer(motionBlurFramebuffer, { motionBlurTexture });
		createFramebuffer(brightPassFramebuffer, { brightPassTexture });
		createFramebuffer(brightPassHalfFramebuffer, { brightHalf });
		createFramebuffer(brightPassQuarterFramebuffer, { brightQuarter });
		createFramebuffer(brightPassOneEightsFramebuffer, { brightOneEights });
		createFramebuffer(gaussianBlurHalfFramebuffer, { gaussianBlurHalfTexture });
		createFramebuffer(gaussianBlurQuarterFramebuffer, { gaussianBlurQuarterTexture });
		createFramebuffer(gaussianBlurOneEightsFramebuffer, { gaussianBlurOneEightsTexture });

		createFramebuffer(gaussianBlurHalfFramebuffer2, { gaussianBlurHalfTexture2 });
		createFramebuffer(gaussianBlurQuarterFramebuffer2, { gaussianBlurQuarterTexture2 });
		createFramebuffer(gaussianBlurOneEightsFramebuffer2, { gaussianBlurOneEightsTexture2 });
	}

	void SparkRenderer::createFramebuffer(GLuint& framebuffer, const std::vector<GLuint>&& colorTextures,
		GLuint renderbuffer)
	{
		glCreateFramebuffers(1, &framebuffer);
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

		std::vector<GLenum> colorAttachments;
		colorAttachments.reserve(colorTextures.size());
		for(unsigned int i = 0; i < colorTextures.size(); ++i)
		{
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorTextures[i], 0);
			colorAttachments.push_back(GL_COLOR_ATTACHMENT0 + i);
		}
		glDrawBuffers(static_cast<GLsizei>(colorAttachments.size()), colorAttachments.data());

		if (renderbuffer != 0)
		{
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderbuffer);
		}

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		{
			throw std::exception("Framebuffer incomplete!");
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void SparkRenderer::addMeshDataToBuffer(const std::vector<glm::vec3>& vertices, const std::vector<GLuint>& indices)
	{
		bufer.addMesh(vertices, indices);
	}

	void SparkRenderer::pushMeshIntoInstancedQueue(ModelMesh* modelMesh)
	{
		std::lock_guard<std::mutex> m(instancedQueueMutex);
		renderInstancedQueue.push_back(modelMesh);
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

	void SparkRenderer::cleanup()
	{
		deleteFrameBuffersAndTextures();
		rendering = false;
		sortingMeshes.join();
	}

	void SparkRenderer::deleteFrameBuffersAndTextures() const
	{
		GLuint textures[9] = { colorTexture, positionTexture, normalsTexture, roughnessTexture, metalnessTexture, lightColorTexture, postProcessingTexture, motionBlurTexture, brightPassTexture };
		glDeleteTextures(9, textures);

		GLuint brightPassDownScale[3] = { brightHalf, brightQuarter, brightOneEights };
		glDeleteTextures(3, brightPassDownScale);

		GLuint frameBuffers[6] = { mainFramebuffer, lightFrameBuffer, postprocessingFramebuffer, motionBlurFramebuffer, cubemapFramebuffer, brightPassFramebuffer };
		glDeleteFramebuffers(6, frameBuffers);
	}

	ModelMesh* SparkRenderer::popMeshFromInstancedQueue()
	{
		std::lock_guard<std::mutex> m(instancedQueueMutex);
		
		const auto modelMesh = renderInstancedQueue.front();
		renderInstancedQueue.pop_front();
		return modelMesh;
	}

}
