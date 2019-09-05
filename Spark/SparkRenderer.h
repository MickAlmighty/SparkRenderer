#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>

class SparkRenderer
{
private:
	GLFWwindow* window = nullptr;
	
	~SparkRenderer();
	SparkRenderer();
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

