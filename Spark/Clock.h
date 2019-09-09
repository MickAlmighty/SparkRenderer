#pragma once
#include <GLFW/glfw3.h>

class Clock
{
private:
	static double deltaTime;
	Clock() = default;
	~Clock() = default;
public:
	static void tick();
	static double getDeltaTime();
	static int getFPS();
};

