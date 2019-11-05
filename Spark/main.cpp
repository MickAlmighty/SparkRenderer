#include <iostream>

#include "JsonSerializer.h"
#include "Spark.h"
#include "Structs.h"
#include "Logging.h"

int main()
{
	Json::Value initVariables = spark::JsonSerializer::readFromFile("settings.json");

	spark::InitializationVariables variables;
	variables.deserialize(initVariables);
	try
	{
		spark::Spark::setup(variables);
		spark::Spark::run();
		spark::Spark::clean();
	}
	catch (std::exception& e)
	{
        SPARK_ERROR("{}", e.what());
		return 1;
	}
	getchar();
	return 0;
}
