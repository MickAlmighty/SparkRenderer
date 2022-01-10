#pragma once

namespace spark
{
class Clock
{
    public:
    static void tick();
    static double getDeltaTime();
    static double getFPS();
    static void enableFixedDelta(double deltaInSeconds);
    static void disableFixedDelta();

    private:
    inline static double deltaTime{0.0};
    inline static double fixedDelta{0.0};
    inline static bool isDeltaFixed{false};
    Clock() = default;
    ~Clock() = default;
};

}  // namespace spark