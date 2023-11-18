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
constexpr uint BLOCK_SIZE = 64;
static uint iDivUp(uint a, uint b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
static uint cudaGridSize(uint totalSize, uint blockSize)
{
	int dim = iDivUp(totalSize, blockSize);
	return dim == 0 ? 1 : dim;
}

static uint3 cudaGridSize2D(uint2 totalSize, uint blockSize)
{
	uint3 gridDims;
	gridDims.x = iDivUp(totalSize.x, blockSize);
	gridDims.y = iDivUp(totalSize.y, blockSize);

	gridDims.x = gridDims.x == 0 ? 1 : gridDims.x;
	gridDims.y = gridDims.y == 0 ? 1 : gridDims.y;
	gridDims.z = 1;

	return gridDims;
}

static uint3 cudaGridSize3D(uint3 totalSize, uint blockSize)
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
		uint pDims = cudaGridSize((uint)size, BLOCK_SIZE);	\
		Func << <pDims, BLOCK_SIZE >> > (				\
		__VA_ARGS__);									\
		cuSynchronize();								\
	}
