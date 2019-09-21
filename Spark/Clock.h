#ifndef CLOCK_H
#define CLOCK_H

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

#endif