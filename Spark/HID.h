#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

struct Mouse
{
	glm::vec2 direction;
	glm::vec2 screenPosition;
};

class HID
{
	static Mouse mouse;
	HID();
	~HID();
public:
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
};

