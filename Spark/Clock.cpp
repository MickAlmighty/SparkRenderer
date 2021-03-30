#include "Clock.h"
#include "Logging.h"

#include <chrono>

namespace spark
{
double Clock::deltaTime = 0;

void Clock::tick()
{
    using namespace std::chrono;
    using seconds = std::ratio<1, 1>;

    static auto lastTime = high_resolution_clock::now();
    const duration<double, seconds> delta = high_resolution_clock::now() - lastTime;
    deltaTime = delta.count();
#ifdef DEBUG
    constexpr auto maxDeltaPeriod{ 1.0 / 10.0 };
    if(deltaTime > maxDeltaPeriod)
    {
        deltaTime = maxDeltaPeriod;
        SPARK_INFO("Game loop duration was longer than 100ms! Delta time set up to 100ms!");
    }
#endif
    lastTime = high_resolution_clock::now();
}

double Clock::getDeltaTime()
{
    return deltaTime;
}

double Clock::getFPS()
{
    return static_cast<int>(1 / deltaTime);
}

}  // namespace spark