#include "HID.h"
#include <iostream>

#include "Clock.h"
#include "Spark.h"

Mouse HID::mouse{};
std::map<int, int> HID::keyStates;

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
	const int state = glfwGetKey(window, key);
	keyStates[key] = state;
}

void HID::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
	static double lastXpos = Spark::WIDTH * 0.5f, lastYPos = Spark::HEIGHT * 0.5f;
	
	mouse.direction.x = xpos - lastXpos; 
	mouse.direction.y = ypos - lastYPos;
	//mouse.direction *= Clock::getDeltaTime();
	lastXpos = xpos;
	lastYPos = ypos;

	mouse.screenPosition.x = xpos;
	mouse.screenPosition.y = ypos;

#ifdef DEBUG
	//std::cout << "Mouse X: " << xpos << " Mouse Y: " << ypos << std::endl;
	std::cout << "Mouse X_Dir: " << mouse.direction.x << " Mouse Y_Dir: " << mouse.direction.y << std::endl;
#endif
}

void HID::clearStates()
{
	mouse.direction *= 0;
}

bool HID::isKeyPressed(int key)
{
	const auto key_it = keyStates.find(key);
	if(key_it != keyStates.end())
	{
		return key_it->second == GLFW_PRESS;
	}
	return false;
}

bool HID::isKeyReleased(int key)
{
	const auto key_it = keyStates.find(key);
	if (key_it != keyStates.end())
	{
		return key_it->second == GLFW_RELEASE;
	}
	return false;
}

bool HID::isKeyHeld(int key)
{
	const auto key_it = keyStates.find(key);
	if (key_it != keyStates.end())
	{
		return key_it->second == GLFW_REPEAT;
	}
	return false;
}
