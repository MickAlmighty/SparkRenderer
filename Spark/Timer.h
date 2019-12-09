#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <iostream>

#define PROFILING 1
#if PROFILING
#define COMBINE(X, Y) X ## Y
#define COMBINE_NAME(X, Y)  COMBINE(X, Y)// helper macro
#define PROFILE_SCOPE(name) Timer COMBINE_NAME(timer, __LINE__)(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCSIG__)
#else
#define PROFILE_SCOPE(name)
#endif

namespace spark {
	
	class Timer
	{
	public:
		static bool capture;

		Timer(const char* measurementName);
		~Timer();

		void stop();
	private:
		const char* name;
		std::chrono::time_point<std::chrono::steady_clock> startTime;
		bool stopped{ false };
	};
}

#endif