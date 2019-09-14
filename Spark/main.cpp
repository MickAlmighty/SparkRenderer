#include <iostream>
#include <Spark.h>
#include <Structs.h>
#include <fstream>
#include <json/writer.h>
#include <json/reader.h>
#include <iomanip>
#include <JsonSerializer.h>

int main()
{
	Json::Value initVariables = JsonSerializer::readFromFile("settings.json");
	
	InitializationVariables variables;
	variables.width = initVariables.get("width", 1280).asInt();
	variables.height = initVariables.get("height", 720).asInt();
	variables.pathToModels = initVariables["pathToModels"].asString();
	variables.pathToResources = initVariables["pathToResources"].asString();

	try
	{
		Spark::setup(variables);
		Spark::run();
		Spark::clean();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return 1;
	}
	getchar();
	return 0;
}
