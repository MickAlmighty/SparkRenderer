#include "Clock.h"

#ifdef DEBUG
#include "Logging.h"
#endif

#include <chrono>

namespace spark
{
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
    if(isDeltaFixed)
        return fixedDelta;

    return deltaTime;
}

double Clock::getFPS()
{
    return 1.0 / getDeltaTime();
}

void Clock::disableFixedDelta()
{
    isDeltaFixed = false;
}

void Clock::enableFixedDelta(double deltaInSeconds)
{
    isDeltaFixed = true;
    deltaInSeconds < 0.0 ? fixedDelta = 0.0 : fixedDelta = deltaInSeconds;
}
}  // namespace spark