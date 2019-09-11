#include "Spark.h"
#include "SparkRenderer.h"
#include "Clock.h"
#include <iostream>
#include "HID.h"
#include "ResourceManager.h"

unsigned int Spark::WIDTH, Spark::HEIGHT;
std::filesystem::path Spark::pathToModels;
std::filesystem::path Spark::pathToResources;

void Spark::setup(InitializationVariables& variables)
{
	WIDTH = variables.width;
	HEIGHT = variables.height;
	pathToModels = variables.pathToModels;
	pathToResources = variables.pathToResources;
	
	SparkRenderer::initOpenGL();
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
	SparkRenderer::getInstance()->cleanup();
}
