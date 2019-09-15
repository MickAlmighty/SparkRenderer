#include <EngineSystems/SparkRenderer.h>
#include <EngineSystems/ResourceManager.h>
#include <EngineSystems/SceneManager.h>
#include <Spark.h>
#include <Clock.h>
#include <iostream>
#include <HID.h>


unsigned int Spark::WIDTH, Spark::HEIGHT;
std::filesystem::path Spark::pathToModelMeshes;
std::filesystem::path Spark::pathToResources;
bool Spark::runProgram = true;

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
	while(SparkRenderer::isWindowOpened() && runProgram)
	{
		Clock::tick();
		glfwPollEvents();
		SceneManager::getInstance()->update();
		SparkRenderer::getInstance()->renderPass();
		HID::clearStates();
#ifdef DEBUG
		std::cout << Clock::getFPS() << std::endl;
#endif
	}
}

void Spark::clean()
{
	SparkRenderer::getInstance()->cleanup();
	SceneManager::getInstance()->cleanup();
	ResourceManager::getInstance()->cleanup();
}
