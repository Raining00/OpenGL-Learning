#pragma once
#include <list>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <algorithm>

#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <limits>

constexpr float EPSILON = std::numeric_limits<float>::epsilon();
constexpr float REAL_MAX = (std::numeric_limits<float>::max)();
constexpr float REAL_MIN = (std::numeric_limits<float>::min)();
constexpr float REAL_EPSILON = (std::numeric_limits<float>::epsilon)();
constexpr float REAL_EPSILON_SQUARED = REAL_EPSILON * REAL_EPSILON;
constexpr unsigned int BLOCK_SIZE = 64;
static unsigned int iDivUp(unsigned int a, unsigned int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
static unsigned int cudaGridSize(unsigned int totalSize, unsigned int blockSize)
{
	int dim = iDivUp(totalSize, blockSize);
	return dim == 0 ? 1 : dim;
}

static uint3 cudaGridSize2D(uint2 totalSize, unsigned int blockSize)
{
	uint3 gridDims;
	gridDims.x = iDivUp(totalSize.x, blockSize);
	gridDims.y = iDivUp(totalSize.y, blockSize);

	gridDims.x = gridDims.x == 0 ? 1 : gridDims.x;
	gridDims.y = gridDims.y == 0 ? 1 : gridDims.y;
	gridDims.z = 1;

	return gridDims;
}

static uint3 cudaGridSize3D(uint3 totalSize, unsigned int blockSize)
{
	uint3 gridDims;
	gridDims.x = iDivUp(totalSize.x, blockSize);
	gridDims.y = iDivUp(totalSize.y, blockSize);
	gridDims.z = iDivUp(totalSize.z, blockSize);

	gridDims.x = gridDims.x == 0 ? 1 : gridDims.x;
	gridDims.y = gridDims.y == 0 ? 1 : gridDims.y;
	gridDims.z = gridDims.z == 0 ? 1 : gridDims.z;

	return gridDims;
}

static uint3 cudaGridSize3D(uint3 totalSize, uint3 blockSize)
{
	uint3 gridDims;
	gridDims.x = iDivUp(totalSize.x, blockSize.x);
	gridDims.y = iDivUp(totalSize.y, blockSize.y);
	gridDims.z = iDivUp(totalSize.z, blockSize.z);

	gridDims.x = gridDims.x == 0 ? 1 : gridDims.x;
	gridDims.y = gridDims.y == 0 ? 1 : gridDims.y;
	gridDims.z = gridDims.z == 0 ? 1 : gridDims.z;

	return gridDims;
}

#define cuSynchronize()	{						\
		char str[200];							\
		cudaDeviceSynchronize();				\
		cudaError_t err = cudaGetLastError();	\
		if (err != cudaSuccess)					\
		{										\
			sprintf(str, "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__);		\
			throw std::runtime_error(std::string(str));																\
		}																											\
	}

#define cuExecute(size, Func, ...){						\
		unsigned int pDims = cudaGridSize((unsigned int)size, BLOCK_SIZE);	\
		Func << <pDims, BLOCK_SIZE >> > (				\
		__VA_ARGS__);									\
		cuSynchronize();								\
	}

static const char* _cudaGetErrorEnum(cudaError_t error) {
	return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char* errorMessage, const char* file,
	const int line) {
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
		fprintf(stderr,
			"%s(%i) : getLastCudaError() CUDA error :"
			" %s : (%d) %s.\n",
			file, line, errorMessage, static_cast<int>(err),
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}