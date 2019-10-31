#ifndef SPARK_H
#define	SPARK_H

#include "Structs.h"
#include "Structs.h"
#include "GUI/SparkGui.h"

#include <filesystem>
#include <GLFW/glfw3.h>

namespace spark {

class Spark
{
public:
	inline static unsigned int WIDTH {1280};
	inline static unsigned int HEIGHT {720};
	inline static std::filesystem::path pathToModelMeshes;
	inline static std::filesystem::path pathToResources;
	inline static GLFWwindow* window = nullptr;
	inline static bool runProgram = true;
	inline static float maxAnisotropicFiltering = 1.0f;
	
	static void setup(InitializationVariables& variables);
	static void run();
	static void resizeWindow(GLuint width, GLuint height);
	static void clean();

private:
	inline static SparkGui sparkGui{};

	~Spark() = default;
	Spark() = default;

	static void initOpenGL();
	static void destroyOpenGLContext();
};
}
#endif