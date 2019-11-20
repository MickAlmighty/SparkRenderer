#ifndef DEVICE_MEMORY_H
#define DEVICE_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <future>

namespace spark {
	namespace cuda {
		template<typename T>
		class DeviceMemory
		{
			DeviceMemory(std::size_t bytes)
			{
				cudaMalloc(&ptr, bytes);
				//cudaMallocHost(&ptr, bytes);
			}
		public:
			T* ptr = nullptr;
			static DeviceMemory AllocateElements(std::size_t n) { return { n * sizeof(T) }; }
			static DeviceMemory AllocateBytes(std::size_t bytes) { return { bytes }; }
			~DeviceMemory()
			{
				if (ptr)
				{
					cudaFree(ptr);
					//const auto future = std::async(std::launch::async, cudaFreeHost, ptr);
					//const auto future = std::async(std::launch::async, cudaFree, ptr);
				}
			}
		};
	}
}
#endif
