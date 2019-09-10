#include "HID.h"
#include <iostream>

#include "Clock.h"
//#include "SparkRenderer.h"

Mouse HID::mouse{};

HID::HID()
{
}


HID::~HID()
{
}

void HID::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	static glm::vec3 cameraPos(0);
	float cameraSpeed = 0.05f; // dopasuj do swoich potrzeb  
	//if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	//	SparkRenderer::getInstance()->camera->moveFront();
	//if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	//	SparkRenderer::getInstance()->camera->moveBack();
	//if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	//	SparkRenderer::getInstance()->camera->moveLeft();
	//if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	//	SparkRenderer::getInstance()->camera->moveRight();
	//SparkRenderer::getInstance()->camera->setPos();
}

void HID::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
	static double lastXpos, lastYPos;
	
	mouse.direction.x = xpos - lastXpos; 
	mouse.direction.y = ypos - lastYPos;
	//mouse.direction *= Clock::getDeltaTime();

	mouse.screenPosition.x = xpos;
	mouse.screenPosition.y = ypos;

	lastXpos = xpos;
	lastYPos = ypos;

#ifdef DEBUG
	//std::cout << "Mouse X: " << xpos << " Mouse Y: " << ypos << std::endl;
	std::cout << "Mouse X_Dir: " << mouse.direction.x << " Mouse Y_Dir: " << mouse.direction.y << std::endl;
#endif
}

void HID::clearStates()
{
	mouse.direction *= 0;
}
