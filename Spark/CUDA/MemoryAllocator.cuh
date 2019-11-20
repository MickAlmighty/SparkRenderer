#ifndef MEMORY_ALLOCATOR_CUH
#define MEMORY_ALLOCATOR_CUH

#include <cuda_runtime.h>
#include <cstdlib>

namespace spark {
	namespace cuda {
		
		class MemoryAllocator
		{
			void* memoryStack = nullptr;
			int size{ 0 };
			void* stackPointer = nullptr;
		
		public:
			__device__ MemoryAllocator(int byteCount)
			{
				memoryStack = malloc(byteCount);
				stackPointer = memoryStack;
				size = byteCount;
			}

			__device__ ~MemoryAllocator()
			{
				free(memoryStack);
			}

			template <typename T>
			__device__ T* allocate()
			{
				T toCopy = T();
				T* object = (T*)stackPointer;
				memcpy(object, &toCopy, sizeof(T));
				stackPointer = object + 1;
				return object;
			}

			template <typename T, typename ... Types>
			__device__ T* allocate(Types ... args)
			{
				T toCopy = T(args...);
				T* object = (T*)stackPointer;
				memcpy(object, &toCopy, sizeof(T));
				stackPointer = object + 1;
				return object;
			}
		};
	}
}

#endif