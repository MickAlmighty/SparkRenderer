#pragma once

#include <array>
#include <functional>

#include "glad_glfw3.h"

namespace spark::profiling
{
template<unsigned int MEASUREMENTS_COUNT>
class GpuTimer
{
    public:
    GpuTimer()
    {
        glGenQueries(MEASUREMENTS_COUNT * 2, queries);
    }

    ~GpuTimer()
    {
        glDeleteQueries(MEASUREMENTS_COUNT * 2, queries);
    }

    template<class F, class... Args>
    void measure(unsigned index, F&& func, Args&&... args);

    std::array<double, MEASUREMENTS_COUNT> getMeasurementsInUs();

    private:
    GLuint queries[MEASUREMENTS_COUNT * 2]{};
    bool processedMeasurements[MEASUREMENTS_COUNT]{};
};

template<unsigned MEASUREMENTS_COUNT>
template<class F, class... Args>
void GpuTimer<MEASUREMENTS_COUNT>::measure(unsigned index, F&& func, Args&&... args)
{
    glQueryCounter(queries[index * 2], GL_TIMESTAMP);
    std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
    glQueryCounter(queries[index * 2 + 1], GL_TIMESTAMP);
    processedMeasurements[index] = true;
}

template<unsigned MEASUREMENTS_COUNT>
std::array<double, MEASUREMENTS_COUNT> GpuTimer<MEASUREMENTS_COUNT>::getMeasurementsInUs()
{
    int lastProcessedIndex{-1};
    for(int i = 0; i < MEASUREMENTS_COUNT; ++i)
    {
        if(processedMeasurements[i])
        {
            lastProcessedIndex = i;
        }
    }

    if(lastProcessedIndex == -1)
        return {};

    GLint done{false};
    while(!done)
    {
        glGetQueryObjectiv(queries[lastProcessedIndex * 2 + 1], GL_QUERY_RESULT_AVAILABLE, &done);
    }

    std::array<double, MEASUREMENTS_COUNT> measurements{};

    for(unsigned int i = 0; i < MEASUREMENTS_COUNT; ++i)
    {
        if(processedMeasurements[i])
        {
            processedMeasurements[i] = false;
            GLuint64 timerStart, timerEnd;
            glGetQueryObjectui64v(queries[i * 2], GL_QUERY_RESULT, &timerStart);
            glGetQueryObjectui64v(queries[i * 2 + 1], GL_QUERY_RESULT, &timerEnd);

            measurements[i] = (timerEnd - timerStart) / 1'000.0;
        }
    }

    return measurements;
}
}  // namespace spark::profiling