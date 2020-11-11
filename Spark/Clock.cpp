#include "Clock.h"
#include "Logging.h"

#include <GLFW/glfw3.h>

namespace spark
{
double Clock::deltaTime = 0;

void Clock::tick()
{
    static double lastTime;
    deltaTime = glfwGetTime() - lastTime;
#ifdef DEBUG
    if(deltaTime > 1.0f / 10.0f)
    {
        deltaTime = 1.0f / 10.0f;
        SPARK_INFO("Game loop duration was longer than 100ms! Delta time set up to 100ms!");
    }
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

}  // namespace spark