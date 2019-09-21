#pragma once
#include <iostream>
#include <Spark.h>
#include <Structs.h>
#include <JsonSerializer.h>

int main()
{
	Json::Value initVariables = JsonSerializer::readFromFile("settings.json");
	
	InitializationVariables variables;
	variables.deserialize(initVariables);
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
