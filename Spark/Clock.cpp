#include "Clock.h"

double Clock::deltaTime = 0;

void Clock::tick()
{
	static double lastTime;
#ifdef DEBUG
	deltaTime > 1.0f / 30.0f ? deltaTime = 1.0f / 30.0f : deltaTime = glfwGetTime() - lastTime;
#else
	deltaTime = glfwGetTime() - lastTime;
#endif
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
