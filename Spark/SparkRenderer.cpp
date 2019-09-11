#include "SparkRenderer.h"
#include <exception>
#include <iostream>
#include "ResourceLoader.h"
#include "HID.h"
#include "ResourceManager.h"
#include "Spark.h"

GLFWwindow* SparkRenderer::window = nullptr;

SparkRenderer* SparkRenderer::getInstance()
{
	static SparkRenderer* spark_renderer = nullptr;
	if(spark_renderer == nullptr)
	{
		spark_renderer = new SparkRenderer();
	}
	return  spark_renderer;
}

void SparkRenderer::setup()
{
	initMembers();
}

void SparkRenderer::initOpenGL()
{
	if (!glfwInit())
	{
		throw std::exception("glfw init failed");
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfwSetErrorCallback(error_callback);

	window = glfwCreateWindow(Spark::WIDTH, Spark::HEIGHT, "Spark", NULL, NULL);
	if (!window)
	{
		throw std::exception("Window creation failed");
	}

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw std::exception("Failed to initialize OpenGL loader!");
	}

	glfwSetKeyCallback(window, HID::key_callback);
	glfwSetCursorPosCallback(window, HID::cursor_position_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	unsigned char pixels[16 * 16 * 4];
	memset(pixels, 0xff, sizeof(pixels));
	GLFWimage image;
	image.width = 16;
	image.height = 16;
	image.pixels = pixels;
	GLFWcursor* cursor = glfwCreateCursor(&image, 0, 0);

	glfwSetCursor(window, cursor);

	glfwSwapInterval(0);
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
	mainShader = ResourceManager::getInstance()->getShader(DEFAULT_SHADER);
	screenShader = ResourceManager::getInstance()->getShader(SCREEN_SHADER);
	postprocessingShader = ResourceManager::getInstance()->getShader(POSTPROCESSING_SHADER);
	model = ResourceManager::getInstance()->findModel(R"(C:\Studia\Semestr6\SparkRenderer\res\models\box\box.obj)");
	camera = new Camera(glm::vec3(0, 0, 2));

	glCreateFramebuffers(1, &mainFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, mainFramebuffer);
	createTexture(colorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);

	GLuint renderbuffer;
	glCreateRenderbuffers(1, &renderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, Spark::WIDTH, Spark::HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderbuffer);

	GLenum attachments[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, attachments);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		throw std::exception("Main framebuffer incomplete!");
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glCreateFramebuffers(1, &postprocessingFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, postprocessingFramebuffer);
	createTexture(postProcessingTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, postProcessingTexture, 0);
	GLenum attachments2[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, attachments2);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		throw std::exception("Postprocessing framebuffer incomplete!");
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SparkRenderer::renderPass()
{
	camera->ProcessKeyboard();
	camera->ProcessMouseMovement(HID::mouse.direction.x, -HID::mouse.direction.y);
	glBindFramebuffer(GL_FRAMEBUFFER, mainFramebuffer);
	glClearColor(0, 0, 0, 1);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	glm::mat4 view = camera->GetViewMatrix();
	glm::mat4 projection = glm::perspective(glm::radians(60.0f), (float)Spark::WIDTH / Spark::HEIGHT, 0.1f, 100.0f);
	
	mainShader->use();
	model->transform.setRotationDegrees(0, 30, 0);
	glm::mat4 MVP = projection * view * model->transform.getMatrix();
	mainShader->setMat4("MVP", MVP);
	model->draw();

	postprocessingPass();
	renderToScreen();
	glfwSwapBuffers(window);
}

void SparkRenderer::postprocessingPass()
{
	glBindFramebuffer(GL_FRAMEBUFFER, postprocessingFramebuffer);
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	postprocessingShader->use();
	postprocessingShader->setVec2("inversedScreenSize", { 1.0f / Spark::WIDTH, 1.0f / Spark::HEIGHT });

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, colorTexture);

	screenQuad.draw();
}


void SparkRenderer::renderToScreen()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	screenShader->use();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, postProcessingTexture);

	screenQuad.draw();
}

void SparkRenderer::cleanup()
{
	glfwDestroyWindow(window);
}

bool SparkRenderer::isWindowOpened() const
{
	return !glfwWindowShouldClose(window);
}

void SparkRenderer::error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s/n", description);
}
