#pragma once
#include <iostream>
#include <cuda_runtime.h>

#if(defined(__NVCC__))
#	define DYN_ALIGN_16 __align__(16) 
#else
#	define DYN_ALIGN_16 alignas(16)
#endif

template<typename T>
class VectorBase
{
public:
    virtual int size()const = 0;
    virtual T& operator[] (unsigned int) = 0;
    virtual const T& operator[] (unsigned int) const = 0;
};


template <typename T, int Dim>
class Vector
{
public:
    __device__ __host__ Vector() {};
    __device__ __host__ ~Vector() {};
};