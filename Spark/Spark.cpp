#include "Spark.h"
#include "SparkRenderer.h"
#include "Clock.h"
#include <iostream>
#include "HID.h"
#include "ResourceManager.h"

void Spark::setup()
{
	SparkRenderer::getInstance()->initOpengl();
	ResourceManager::getInstance()->loadResources();

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
		HID::clearStates();
	}
}

void Spark::clean()
{
}
