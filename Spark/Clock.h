#pragma once

namespace spark
{
class Clock
{
    private:
    static double deltaTime;
    Clock() = default;
    ~Clock() = default;

    public:
    static void tick();
    static double getDeltaTime();
    static double getFPS();
};

}  // namespace spark