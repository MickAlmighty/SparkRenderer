#ifndef MEMORY_ALLOCATOR_CUH
#define MEMORY_ALLOCATOR_CUH

#include <cuda_runtime.h>
#include <cstdlib>

namespace spark {
	namespace cuda {
		
		class MemoryAllocator
		{
			void* memoryPool = nullptr;
			int size{ 0 };
		
		public:
			__device__ MemoryAllocator() {}
			
			__device__ MemoryAllocator(int byteCount)
			{
				memoryPool = malloc(byteCount);
				size = byteCount;
			}

			__device__ ~MemoryAllocator()
			{
				if(memoryPool)
					free(memoryPool);
			}

			template <typename T>
			__device__ T* ptr(size_t offsetFromBeginning) const
			{
				return static_cast<T*>(memoryPool) + offsetFromBeginning;
			}
		};
	}
}

#endif