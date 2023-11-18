#pragma once
#include <cuda_runtime.h>
#define CUDA_BACKEND

typedef unsigned int uint;
/**
 * @brief Be aware do not use this structure on GPU if the data size is large.
 * 
 * @tparam T 
 */
template <typename T>
class STLBuffer
{
public:
    using iterator = T * ;

    __device__ __host__  STLBuffer() {};
    
    __device__ __host__ void reserve(T* beg, size_t buffer_size) {
        m_startLoc = beg;
        m_maxSize = buffer_size;
    }

    __device__ __host__ uint max_size() { return m_maxSize; }

protected:
    __device__ __host__ inline T* bufferEnd() {
        return m_startLoc + m_maxSize;
    }

    uint m_maxSize = 0;
    
    T* m_startLoc = nullptr;
};

template <typename T>
class List : public STLBuffer<T>
{
public:
    using iterator = T*;

    __device__ __host__ List();
    
    __device__ __host__ iterator find(T val);

    __device__ __host__ inline iterator begin() {
        return this->m_startLoc;
    }

    __device__ __host__ inline iterator end(){
        return this->m_startLoc + m_size;
    }

    __device__ inline T& operator [] (uint id) {
        return this->m_startLoc[id];
    }

    __device__ inline T& operator [] (uint id) const {
        return this->m_startLoc[id];
    }

    __device__ __host__ void assign(T* beg, int num, int buffer_size) {
        this->m_startLoc = beg;
        m_size = num;
        this->m_maxSize = buffer_size;
    }

    __device__ __host__ void clear();

    __device__ __host__ uint size();

    __device__ __host__ inline iterator insert(T val);

#ifdef CUDA_BACKEND
		__device__ inline iterator atomicInsert(T val);
#endif

    __device__ __host__ inline T front();
    __device__ __host__ inline T back();

    __device__ __host__ bool empty();

private:
    uint m_size = 0;
};

template <typename T>
__device__ __host__ List<T>::List()
{
}

template <typename T>
__device__ __host__ T* List<T>::find(T val)
{
    return nullptr;
}


template <typename T>
__device__ __host__ T* List<T>::insert(T val)
{
    //return nullptr if the data buffer is full
    if (m_size >= this->m_maxSize) return nullptr;

    this->m_startLoc[m_size] = val;
    m_size++;

    return this->m_startLoc + m_size - 1;;
}

#ifdef CUDA_BACKEND
	template <typename T>
	__device__ T* List<T>::atomicInsert(T val)
	{
		//return nullptr if the data buffer is full
		if (m_size >= this->m_maxSize) return nullptr;
		
		
		int index = atomicAdd(&(this->m_size), 1+(val-val));
		//int index = atomicAdd(&(this->m_size), 1);
		//int index = 0;//Onlinux platform, this is a bug, not yet resolved.

		this->m_startLoc[index] = val;

		return this->m_startLoc + index;
	}
#endif

template <typename T>
__device__ __host__ void List<T>::clear()
{
    m_size = 0;
}

template <typename T>
__device__ __host__ uint List<T>::size()
{
    return m_size;
}

template <typename T>
__device__ __host__ bool List<T>::empty()
{
    return this->m_startLoc == nullptr;
}

template <typename T>
__device__ __host__ T List<T>::front()
{
    return this->m_startLoc[0];
}

template <typename T>
__device__ __host__ T List<T>::back()
{
    return this->m_startLoc[m_size - 1];
}