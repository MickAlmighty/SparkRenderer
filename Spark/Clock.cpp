#include <Clock.h>
#include <GLFW/glfw3.h>
double Clock::deltaTime = 0;

void Clock::tick()
{
	static double lastTime;
	deltaTime = glfwGetTime() - lastTime;
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
