#include <iostream>
#include "Spark.h"

int main()
{
	try
	{
		Spark::setup();
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
