#ifndef BINARY_HEAP_CUH
#define BINARY_HEAP_CUH

#include <cuda_runtime_api.h>
#include <nvfunctional>

namespace spark {
	namespace cuda {
		template <typename T>
		class BinaryHeap
		{
		public:
			unsigned int size{ 0 };
			nvstd::function<bool(const T& lhs, const T& rhs)> comparator{ [] __device__(const T& lhs, const T& rhs)
			{
				return lhs < rhs;
			} };
			T* array = nullptr;

			__device__ BinaryHeap(T* array_) : array(array_) {}
			__device__ BinaryHeap(T* array_, const nvstd::function<bool(const T& lhs, const T& rhs)>& comp) : array(array_), comparator(comp) {}
			__device__ ~BinaryHeap() {}
			__device__ void insert(const T& key);
			__device__ T pop_front();
			__device__ T removeValue(int index);
			__device__ int findIndex(const T& value);
			__device__ int findIndex_if(const nvstd::function<bool(const T& value)>& isEqual);
		private:
			__device__ int parent(int index) const;
			__device__ int left(int index) const;
			__device__ int right(int index) const;
			__device__ void swap(T& lhs, T& rhs) const noexcept;
		};

		template <typename T>
		__device__ int BinaryHeap<T>::parent(int index) const
		{
			return (index - 1) / 2;
		}

		template <typename T>
		__device__ int BinaryHeap<T>::left(int index) const
		{
			return index * 2 + 1;
		}

		template <typename T>
		__device__ int BinaryHeap<T>::right(int index) const
		{
			return index * 2 + 2;
		}

		template <typename T>
		__device__ void BinaryHeap<T>::swap(T& lhs, T& rhs) const noexcept
		{
			T tmp = lhs;
			lhs = rhs;
			rhs = tmp;
		}

		template <typename T>
		__device__ void BinaryHeap<T>::insert(const T& key)
		{
			array[size] = key;
			int parentIndex = parent(size);
			int keyIndex = size;
			while (true)
			{
				if (keyIndex == parentIndex)
					break;
				if (comparator(array[keyIndex], array[parentIndex]))
				{
					swap(array[keyIndex], array[parentIndex]);
				}
				else
					break;
				keyIndex = parentIndex;
				parentIndex = parent(parentIndex);
			}
			++size;
		}

		template <typename T>
		__device__ T BinaryHeap<T>::pop_front()
		{
			return removeValue(0);
		}

		template <typename T>
		__device__ T BinaryHeap<T>::removeValue(int index)
		{
			if (size == 0)
				return {};

			if (size == 1)
			{
				size -= 1;
				const T value = array[size];
				//memset(array + size, 0, sizeof(T));
				return value;
			}

			swap(array[index], array[size - 1]);
			
			size -= 1;
			int keyIndex = index;
			while (true)
			{
				const int leftIndex = left(keyIndex);
				int rightIndex = leftIndex + 1;

				if (leftIndex >= size)
					break;

				if (rightIndex >= size)
				{
					if (comparator(array[leftIndex], array[keyIndex]))
					{
						swap(array[leftIndex], array[keyIndex]);
					}
					break;
				}

				//it means both leftIndex and rightIndex are < size
				if (comparator(array[leftIndex], array[keyIndex]))
				{
					if (comparator(array[leftIndex], array[rightIndex]))
					{
						swap(array[leftIndex], array[keyIndex]);
						keyIndex = leftIndex;
					}
					else
					{
						swap(array[rightIndex], array[keyIndex]);
						keyIndex = rightIndex;
					}
					continue;
				}

				if (comparator(array[rightIndex], array[keyIndex]))
				{
					swap(array[rightIndex], array[keyIndex]);
					keyIndex = rightIndex;
					continue;
				}
				break;
			}

			const T value = array[size];
			//memset(array + size, 0, sizeof(T));
			return value;
		}

		template <typename T>
		__device__ int BinaryHeap<T>::findIndex(const T& value)
		{
			for(int i = 0; i < size; ++i)
			{
				if (array[i] == value)
				{
					return i;
				}
			}
			return -1;
		}

		template <typename T>
		__device__ int BinaryHeap<T>::findIndex_if(const nvstd::function<bool(const T& value)>& isEqual)
		{
			for (int i = 0; i < size; ++i)
			{
				if (isEqual(array[i]))
				{
					return i;
				}
			}
			return -1;
		}
	}
}
#endif