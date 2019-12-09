#include <iostream>

#include "JsonSerializer.h"
#include "Spark.h"
#include "Structs.h"
#include "Timer.h"
#include "ProfilingWriter.h"

int main()
{
	Json::Value initVariables = spark::JsonSerializer::readFromFile("settings.json");

	spark::InitializationVariables variables;
	variables.deserialize(initVariables);
	try
	{
		spark::ProfilingWriter::get().beginSession("test");
		spark::Spark::setup(variables);
		spark::Spark::run();
		spark::Spark::clean();
		spark::ProfilingWriter::get().endSession();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
		getchar();
		return 1;
	}
	return 0;
}
