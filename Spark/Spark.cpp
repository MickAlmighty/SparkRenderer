#include "Spark.h"
#include "SparkRenderer.h"
#include "Clock.h"
#include <iostream>

void Spark::setup()
{
	SparkRenderer::getInstance()->setup();
}

void Spark::run()
{
	
	while(SparkRenderer::getInstance()->isWindowOpened())
	{
		Clock::tick();
		glfwPollEvents();
		SparkRenderer::getInstance()->renderPass();
#ifdef DEBUG
		std::cout << Clock::getFPS() << std::endl;
#endif
	}
}

void Spark::clean()
{
}
