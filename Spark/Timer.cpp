#include "Timer.h"
#include <future>

namespace spark {
	std::map<std::string, double> Timer::measurements{};

	double Timer::getMeasurement(const std::string&& measurementName)
	{
		const auto measurementIt = measurements.find(measurementName);
		if (measurementIt != std::end(measurements))
		{
			return measurementIt->second;
		}
		return -1.0;
	}

	Timer::Timer(const std::string&& measurementName)
	{
		name = measurementName;
		startTime = std::chrono::high_resolution_clock::now();
	}

	Timer::~Timer()
	{
		const auto t2 = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> duration = t2 - startTime;
		measurements[name] = duration.count();

		const auto future = std::async(std::launch::async, [this, &duration]()
		{
			std::cout << name.c_str() << ", duration: " << duration.count() << " ms" << std::endl;
		});
	}
}
