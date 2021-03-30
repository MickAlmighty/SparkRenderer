#pragma once

namespace spark
{
class Clock
{
    public:
    static void tick();
    static double getDeltaTime();
    static double getFPS();

    private:
    static double deltaTime;
    Clock() = default;
    ~Clock() = default;
};

}  // namespace spark