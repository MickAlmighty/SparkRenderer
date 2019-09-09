#include "SparkRenderer.h"
#include <exception>
#include <iostream>

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
	initOpengl();
	initMembers();
}

void SparkRenderer::initOpengl()
{
	if (!glfwInit())
	{
		throw std::exception("glfw init failed");
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfwSetErrorCallback(error_callback);

	window = glfwCreateWindow(640, 480, "Spark", NULL, NULL);
	if (!window)
	{
		throw std::exception("Window creation failed");
	}

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw std::exception("Failed to initialize OpenGL loader!");
	}

	glfwSwapInterval(0);
}

void SparkRenderer::initMembers()
{
	screenQuad.setup();
	shader = std::make_unique<Shader>("C:/Studia/Semestr6/SparkRenderer/shaders/default.vert", "C:/Studia/Semestr6/SparkRenderer/shaders/default.frag");
}

void SparkRenderer::renderPass()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(0, 0, 0, 1);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	shader->use();

	screenQuad.draw();

	glfwSwapBuffers(window);
	glfwPollEvents();
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
	fprintf(stderr, "Error: %s\n", description);
}
