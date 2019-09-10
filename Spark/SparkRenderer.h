#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Structs.h"
#include "Shader.h"
#include "Model.h"
#include "Camera.h"

class SparkRenderer
{
private:
	ScreenQuad screenQuad{};
	Model* model = nullptr;
	
	~SparkRenderer() = default;
	SparkRenderer() = default;
	
	void initMembers();
	std::unique_ptr<Shader> shader = nullptr;
public:
	static GLFWwindow* window;
	static SparkRenderer* getInstance();
	SparkRenderer(const SparkRenderer&) = delete;
	SparkRenderer operator=(const SparkRenderer&) = delete;
	Camera* camera = nullptr;
	void initOpengl();

	void setup();
	void renderPass();
	void cleanup();

	bool isWindowOpened() const;

	static void error_callback(int error, const char* description);
};

