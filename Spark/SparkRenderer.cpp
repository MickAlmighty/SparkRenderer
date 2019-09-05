#include "SparkRenderer.h"
#include <exception>
#include <iostream>


SparkRenderer::SparkRenderer()
{
}


SparkRenderer::~SparkRenderer()
{
}

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

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

	if(!glfwInit())
	{
		throw std::exception("glfw init failed");
	}

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

	glfwSwapInterval(1);

}

void SparkRenderer::renderPass()
{
	glfwPollEvents();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(0, 0, 0, 1);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	
	glfwSwapBuffers(window);
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
