#ifndef SPARK_H
#define	SPARK_H
#include <filesystem>
#include <Structs.h>

class Spark
{
private:
	~Spark() = default;
	Spark() = default;
public:
	static unsigned int WIDTH, HEIGHT;
	static std::filesystem::path pathToModelMeshes;
	static std::filesystem::path pathToResources;

	static bool runProgram;

	static void setup(InitializationVariables& variables);
	static void run();
	static void clean();
};

#endif