#pragma once

#include <map>
#include <chrono>
#include <iostream>

namespace spark
{
class Timer
{
    public:
    static inline std::map<std::string, double> measurements;
    static double getMeasurement(const std::string&& measurementName);
    Timer(const std::string&& measurementName);
    ~Timer();

    private:
    std::string name;
    std::chrono::time_point<std::chrono::steady_clock> startTime;
};
}  // namespace spark