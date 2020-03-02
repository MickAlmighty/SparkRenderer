#ifndef CLOCK_H
#define CLOCK_H

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
    static int getFPS();
};

}  // namespace spark
#endif