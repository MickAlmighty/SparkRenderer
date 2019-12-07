#include "Timer.h"

#include <thread>

#include "ProfilingWriter.h"

namespace spark {
	bool Timer::capture = false;

	Timer::Timer(const std::string&& measurementName)
	{
		name = measurementName;
		startTime = std::chrono::high_resolution_clock::now();
	}

	void Timer::stop()
	{
		if (!capture)
			return;
		const auto endTime = std::chrono::high_resolution_clock::now();

		const auto start = std::chrono::time_point_cast<std::chrono::microseconds>(startTime).time_since_epoch().count();
		const auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTime).time_since_epoch().count();

		const auto threadID = static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));

		
		ProfilingWriter::get().writeRecord({ name, start, end, threadID });

		stopped = true;
	}

	Timer::~Timer()
	{
		if(!stopped)
			stop();
	}
}
