#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Structs.h"
#include "Shader.h"
#include "Model.h"

class SparkRenderer
{
private:
	GLFWwindow* window = nullptr;
	ScreenQuad screenQuad{};
	Model* model = nullptr;

	~SparkRenderer() = default;
	SparkRenderer() = default;
	void initOpengl();
	void initMembers();
	std::unique_ptr<Shader> shader = nullptr;
public:
	static SparkRenderer* getInstance();
	SparkRenderer(const SparkRenderer&) = delete;
	SparkRenderer operator=(const SparkRenderer&) = delete;

	void setup();
	void renderPass();
	void cleanup();

	bool isWindowOpened() const;

	static void error_callback(int error, const char* description);
};

