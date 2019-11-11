#ifndef LIST_CUH
#define LIST_CUH

#include <nvfunctional>

namespace spark {
	namespace cuda {
		template <typename T>
		class Iterator
		{
		public:
			T value;
			Iterator<T>* previous = nullptr;
			Iterator<T>* next = nullptr;

			__device__ Iterator<T>(const T& val) : value(val) {}

			__device__ void placePrevious(Iterator<T>* iterator)
			{
				previous = iterator;
				iterator->next = this;
			}

			__device__ void placeNext(Iterator<T>* iterator)
			{
				next = iterator;
				iterator->previous = this;
			}
		};

		template <typename T>
		class List
		{
		public:
			int size = 0;
			Iterator<T>* first = nullptr;
			Iterator<T>* last = nullptr;

			__device__ List() = default;

			__device__ List(nvstd::function<bool(const T& lhs, const T& rhs)>& compare) : comparator(compare) { }

			__device__ ~List<T>()
			{
				for (Iterator<T>* iterator = first; iterator != nullptr;)
				{
					Iterator<T>* toDelete = iterator;
					iterator = iterator->next;
					delete toDelete;
				}
			}

			__device__ Iterator<T>* insert(T value)
			{
				Iterator<T>* it = new Iterator<T>(value);
				if (size == 0)
				{
					first = it;
					++size;
					last = first;
					return it;
				}

				if (size == 1)
				{
					if(comparator(it->value, first->value))
					{
						//it is first now
						it->placeNext(last);
						first = it;
						++size;
					}
					else
					{
						//it is last now
						it->placePrevious(last);
						first = last;
						last = it;
						++size;
					}
					return it;
				}

				for (Iterator<T>* iterator = first; iterator != nullptr; iterator = iterator->next)
				{
					if (comparator(it->value, iterator->value))
					{
						if (iterator == first)
						{
							iterator->placePrevious(it);
							first = it;
							++size;
							return it;
						}

						if (iterator != first)
						{
							iterator->previous->placeNext(it);
							iterator->placePrevious(it);
							++size;
							return it;
						}
					}
				}
				last->placeNext(it);
				last = it;
				++size;
				return it;
			}

			__device__ Iterator<T>* insert(Iterator<T>* it_)
			{
				Iterator<T>* it = it_;
				if (size == 0)
				{
					first = it;
					++size;
					last = first;
					return it;
				}

				if (size == 1)
				{
					if (comparator(it->value, first->value))
					{
						//it is first now
						it->placeNext(last);
						first = it;
						++size;
					}
					else
					{
						//it is last now
						it->placePrevious(last);
						first = last;
						last = it;
						++size;
					}
					return it;
				}

				for (Iterator<T>* iterator = first; iterator != nullptr; iterator = iterator->next)
				{
					if (comparator(it->value, iterator->value))
					{
						if (iterator == first)
						{
							iterator->placePrevious(it);
							first = it;
							++size;
							return it;
						}

						if (iterator != first)
						{
							iterator->previous->placeNext(it);
							iterator->placePrevious(it);
							++size;
							return it;
						}
					}
				}
				last->placeNext(it);
				last = it;
				++size;
				return it;
			}

			__device__ bool remove(const T& value)
			{
				for(auto it = first; it != nullptr; it = it->next)
				{
					if (comparator(value, it->value))
					{
						if (it == first && it == last)
						{
							first = nullptr;
							last = nullptr;
							delete it;
							size = 0;
							return true;
						}

						if (it == first)
						{
							it->next->previous = nullptr;
							first = it->next;
							delete it;
							--size;
							return true;
						}

						if (it == last)
						{
							it->previous->next = nullptr;
							last = it->previous;
							--size;
							delete it;
							return true;
						}
					}
				}
				return false;
			}

			__device__ bool remove(const Iterator<T>* toRemove)
			{
				for (auto it = first; it != nullptr; it = it->next)
				{
					if (toRemove == it)
					{
						if (it == first && it == last)
						{
							first = nullptr;
							last = nullptr;
							delete it;
							size = 0;
							return true;
						}

						if (it == first)
						{
							it->next->previous = nullptr;
							first = it->next;
							delete it;
							--size;
							return true;
						}

						if (toRemove == last)
						{
							it->previous->next = nullptr;
							last = it->previous;
							--size;
							delete it;
						}
					}
				}
				return false;
			}

			__device__ Iterator<T>* find(const T& toFind)
			{
				for (auto it = first; it != nullptr; it = it->next)
				{
					if (toFind == it->value)
					{
						return it;
					}
				}
				return nullptr;
			}

			__device__ Iterator<T>* find_if(nvstd::function<bool(const T& value)>& isEqual)
			{
				for (auto it = first; it != nullptr; it = it->next)
				{
					if (isEqual(it->value))
					{
						return it;
					}
				}
				return nullptr;
			}

			__device__ Iterator<T>* pop_front()
			{
				if (size == 0)
					return nullptr;

				Iterator<T>* pop_front_it = nullptr;

				if (first == last)
				{
					pop_front_it = first;
					first = nullptr;
					last = nullptr;
					pop_front_it->next = nullptr;
					size = 0;
					return pop_front_it;
				}

				pop_front_it = first;
				first->next->previous = nullptr;
				first = first->next;
				pop_front_it->next = nullptr;
				--size;
				return pop_front_it;
			}

			__device__ Iterator<T>* pop_back()
			{
				if (size == 0)
					return nullptr;

				Iterator<T>* pop_back_it = nullptr;

				if (last == first)
				{
					pop_back_it = last;
					first = nullptr;
					last = nullptr;
					pop_back_it->previous = nullptr;
					size = 0;
					return pop_back_it;
				}

				pop_back_it = last;
				last->previous->next = nullptr;
				last = last->previous;
				pop_back_it->previous = nullptr;
				--size;
				return pop_back_it;
			}

		private:
			nvstd::function<bool(const T& lhs, const T& rhs)> comparator = [] __device__ (const T& lhs, const T& rhs) { return lhs < rhs; };
		};

	}
}
#endif