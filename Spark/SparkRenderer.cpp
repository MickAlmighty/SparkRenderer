#include "SparkRenderer.h"
#include <exception>
#include <iostream>
#include "ResourceLoader.h"
#include "HID.h"
#include "ResourceManager.h"

GLFWwindow* SparkRenderer::window = nullptr;

SparkRenderer* SparkRenderer::getInstance()
{
	static SparkRenderer* spark_renderer = nullptr;
	if(spark_renderer == nullptr)
	{
		spark_renderer = new SparkRenderer();
	}
	return  spark_renderer;
}

void SparkRenderer::setup()
{
	initMembers();
}

void SparkRenderer::initOpengl()
{
	if (!glfwInit())
	{
		throw std::exception("glfw init failed");
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfwSetErrorCallback(error_callback);

	window = glfwCreateWindow(1280, 720, "Spark", NULL, NULL);
	if (!window)
	{
		throw std::exception("Window creation failed");
	}

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw std::exception("Failed to initialize OpenGL loader!");
	}

	glfwSetKeyCallback(window, HID::key_callback);
	glfwSetCursorPosCallback(window, HID::cursor_position_callback);
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	unsigned char pixels[16 * 16 * 4];
	memset(pixels, 0xff, sizeof(pixels));
	GLFWimage image;
	image.width = 16;
	image.height = 16;
	image.pixels = pixels;
	GLFWcursor* cursor = glfwCreateCursor(&image, 0, 0);

	glfwSetCursor(window, cursor);

	glfwSwapInterval(0);
}

void SparkRenderer::initMembers()
{
	screenQuad.setup();
	shader = std::make_unique<Shader>("C:/Studia/Semestr6/SparkRenderer/res/shaders/default.vert", "C:/Studia/Semestr6/SparkRenderer/res/shaders/default.frag");
	model = ResourceManager::getInstance()->findModel(R"(C:\Studia\Semestr6\SparkRenderer\res\models\box\box.obj)");
	camera = new Camera(glm::vec3(0, 0, 2));
}

void SparkRenderer::renderPass()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(0, 0, 0, 1);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	//glm::mat4 view = glm::lookAt(glm::vec3(0, 0, -3), glm::vec3(0), glm::vec3(0, 1, 0));
	glm::mat4 view = camera->GetViewMatrix();
	glm::mat4 projection = glm::perspective(glm::radians(70.0f), 1280.0f / 720.0f, 0.1f, 100.0f);
	
	shader->use();
	model->transform.setRotationDegrees(30, 30, 0);
	glm::mat4 MVP = projection * view * model->transform.getMatrix();
	shader->setMat4("MVP", MVP);
	model->draw();

	//screenQuad.draw();

	glfwSwapBuffers(window);
}

void SparkRenderer::cleanup()
{
	glfwDestroyWindow(window);
}

bool SparkRenderer::isWindowOpened() const
{
	return !glfwWindowShouldClose(window);
}

void SparkRenderer::error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s/n", description);
}
