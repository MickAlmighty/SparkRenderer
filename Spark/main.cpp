#include <iostream>
#include "Spark.h"
#include "Structs.h"


int main()
{
	InitializationVariables variables{
		1280,
		720,
		R"(C:\Studia\Semestr6\SparkRenderer\res\models)",
		R"(C:\Studia\Semestr6\SparkRenderer\res)"
	};

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
	return 0;
}
