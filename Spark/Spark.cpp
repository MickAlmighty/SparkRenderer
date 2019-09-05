#include "Spark.h"
#include "SparkRenderer.h"

void Spark::setup()
{
	SparkRenderer::getInstance()->setup();
}

void Spark::run()
{
	
	while(SparkRenderer::getInstance()->isWindowOpened())
	{
		glfwPollEvents();
		SparkRenderer::getInstance()->renderPass();
	}
}

void Spark::clean()
{
}
