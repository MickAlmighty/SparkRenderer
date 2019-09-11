#pragma once
#include <iostream>
#include <filesystem>
#include "Structs.h"

class Spark
{
private:
	~Spark() = default;
	Spark() = default;
public:
	static unsigned int WIDTH, HEIGHT;
	static std::filesystem::path pathToModels;
	static std::filesystem::path pathToResources;
	
	static void setup(InitializationVariables& variables);
	static void run();
	static void clean();
};

