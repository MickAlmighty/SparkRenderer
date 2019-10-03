#include "Clock.h"

#include <GLFW/glfw3.h>

namespace spark {

double Clock::deltaTime = 0;

void Clock::tick()
{
	static double lastTime;
	deltaTime = glfwGetTime() - lastTime;
	if (deltaTime > 1.0f / 60.0f) deltaTime = 1.0f / 60.0f;
	lastTime = glfwGetTime();
}

double Clock::getDeltaTime()
{
	return deltaTime;
}

int Clock::getFPS()
{
	return static_cast<int>(1 / deltaTime);
}

}