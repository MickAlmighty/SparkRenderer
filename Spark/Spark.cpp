#include "Spark.h"
#include "SparkRenderer.h"
#include "Clock.h"
#include <iostream>
#include "HID.h"
#include "ResourceManager.h"
#include "SceneManager.h"

unsigned int Spark::WIDTH, Spark::HEIGHT;
std::filesystem::path Spark::pathToModelMeshes;
std::filesystem::path Spark::pathToResources;

void Spark::setup(InitializationVariables& variables)
{
	WIDTH = variables.width;
	HEIGHT = variables.height;
	pathToModelMeshes = variables.pathToModels;
	pathToResources = variables.pathToResources;
	
	SparkRenderer::initOpenGL();
	ResourceManager::getInstance()->loadResources();
	SceneManager::getInstance()->setup();

	SparkRenderer::getInstance()->setup();
}

void Spark::run()
{
	while(SparkRenderer::isWindowOpened())
	{
		Clock::tick();
		glfwPollEvents();
		SceneManager::getInstance()->update();

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
	SceneManager::getInstance()->cleanup();
	ResourceManager::getInstance()->cleanup();
}
