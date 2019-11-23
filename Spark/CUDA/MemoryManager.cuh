#ifndef MEMORY_MANAGER_CUH
#define MEMORY_MANAGER_CUH

#include <cuda_runtime.h>
#include <cstdlib>

namespace spark {
	namespace cuda {

		class MemoryManager
		{
			void* memoryStack = nullptr;
			void* stackHead = nullptr;
		public:
			__device__ MemoryManager(void* addressPool): memoryStack(addressPool), stackHead(addressPool)
			{

			}

			template <typename T>
			__device__ T* allocate()
			{
				T toCopy;
				T* object = static_cast<T*>(stackHead);
				memcpy(object, &toCopy, sizeof(T));
				stackHead = object + 1;
				return object;
			}

			template <typename T, typename ... Types>
			__device__ T* allocate(Types& ... args)
			{
				T toCopy = T(args...);
				T* object = static_cast<T*>(stackHead);
				memcpy(object, &toCopy, sizeof(T));
				stackHead = object + 1;
				return object;
			}
		};
	}
}

#endif
