#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <map>

struct Mouse
{
	glm::vec2 direction;
	glm::vec2 screenPosition;
};

class HID
{
	HID();
	~HID();
	static std::map<int, int> keyStates;
public:
	static Mouse mouse;
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
	static void clearStates();
	static bool isKeyPressed(int key);
};

