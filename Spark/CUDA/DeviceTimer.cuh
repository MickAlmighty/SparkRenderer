#ifndef DEVICE_TIMER_CUH
#define DEVICE_TIMER_CUH

#include <cuda_runtime_api.h>
#include <ctime>
#include <stdio.h>

namespace spark {
	namespace cuda {
		class DeviceTimer
		{
			clock_t start_time{ 0 };

		public:
			__device__ DeviceTimer()
			{
				start_time = clock();
			}

			__device__ void reset()
			{
				start_time = clock();
			}

			__device__ void printTime(char const* const format) const
			{
				const clock_t end_time = clock();
				printf(format, (end_time - start_time) / 1'253'000.0f);
			}
		};
	}
}

#endif