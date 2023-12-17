#pragma once
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <iostream>
#include <memory>
#include <cmath>
enum DeviceType
{
	CPU,
	GPU,
	UNDEFINED
};

template<typename T, DeviceType deviceType> class Array;

template<typename T>
class Array<T, DeviceType::CPU>
{
public:
	Array()
	{
	};

	Array(unsigned int num)
	{
		mData.resize((size_t)num);
	}

	~Array() {};

	void resize(const unsigned int n);

	/*!
	*	\brief	Clear all data to zero.
	*/
	void reset();

	void clear();

	inline const T*	begin() const { return mData.size() == 0 ? nullptr : &mData[0]; }
	inline T*	begin() { return mData.size() == 0 ? nullptr : &mData[0]; }

	inline const std::vector<T>* handle() const { return &mData; }
	inline std::vector<T>* handle() { return &mData; }

	DeviceType	deviceType() { return DeviceType::CPU; }

	inline T& operator [] (unsigned int id)
	{
		return mData[id];
	}

	inline const T& operator [] (unsigned int id) const
	{
		return mData[id];
	}

	inline unsigned int size() const { return (unsigned int)mData.size(); }
	inline bool isCPU() const { return true; }
	inline bool isGPU() const { return false; }
	inline bool isEmpty() const { return mData.empty(); }

	inline void pushBack(T ele) { mData.push_back(ele); }

	void assign(const T& val);
	void assign(unsigned int num, const T& val);

#ifndef NO_BACKEND
	void assign(const Array<T, DeviceType::GPU>& src);
#endif

	void assign(const Array<T, DeviceType::CPU>& src);

	friend std::ostream& operator<<(std::ostream &out, const Array<T, DeviceType::CPU>& cArray)
	{
		for (unsigned int i = 0; i < cArray.size(); i++)
		{
			out << i << ": " << cArray[i] << std::endl;
		}

		return out;
	}

private:
	std::vector<T> mData;
};

template<typename T>
void Array<T, DeviceType::CPU>::resize(const unsigned int n)
{
	mData.resize(n);
}

template<typename T>
void Array<T, DeviceType::CPU>::clear()
{
	mData.clear();
}

template<typename T>
void Array<T, DeviceType::CPU>::reset()
{
	memset((void*)mData.data(), 0, mData.size()*sizeof(T));
}

template<typename T>
void Array<T, DeviceType::CPU>::assign(const Array<T, DeviceType::CPU>& src)
{
	if (mData.size() != src.size())
		this->resize(src.size());

	memcpy(this->begin(), src.begin(), src.size() * sizeof(T));
}

template<typename T>
void Array<T, DeviceType::CPU>::assign(const T& val)
{
	mData.assign(mData.size(), val);
}

template<typename T>
void Array<T, DeviceType::CPU>::assign(unsigned int num, const T& val)
{
	mData.assign(num, val);
}

template<typename T>
using CArray = Array<T, DeviceType::CPU>;


template<typename T>
void Array<T, DeviceType::CPU>::assign(const Array<T, DeviceType::GPU>& src)
{
	if (mData.size() != src.size())
		this->resize(src.size());

	cudaMemcpy(this->begin(), src.begin(), src.size() * sizeof(T), cudaMemcpyDeviceToHost);
}

/*!
*	\class	Array
*	\brief	This class is designed to be elegant, so it can be directly passed to GPU as parameters.
*/
template<typename T>
class Array<T, DeviceType::GPU>
{
public:
	Array()
	{
	};

	Array(unsigned int num)
	{
		this->resize(num);
	}

	/*!
	*	\brief	Do not release memory here, call clear() explicitly.
	*/
	~Array() {};

	void resize(const unsigned int n);

	/*!
	*	\brief	Clear all data to zero.
	*/
	void reset();

	/*!
	*	\brief	Free allocated memory.	Should be called before the object is deleted.
	*/
	void clear();

	__device__ __host__ inline const T*	begin() const { return mData; }
	__device__ __host__ inline T*	begin() { return mData; }

	DeviceType	deviceType() { return DeviceType::GPU; }

	__device__ inline T& operator [] (unsigned int id) {
		return mData[id];
	}

	__device__ inline T& operator [] (unsigned int id) const {
		return mData[id];
	}

	__device__ __host__ inline unsigned int size() const { return mTotalNum; }
	__device__ __host__ inline bool isCPU() const { return false; }
	__device__ __host__ inline bool isGPU() const { return true; }
	__device__ __host__ inline bool isEmpty() const { return mData == nullptr; }

	void assign(const Array<T, DeviceType::GPU>& src);
	void assign(const Array<T, DeviceType::CPU>& src);
	void assign(const std::vector<T>& src);

	void assign(const Array<T, DeviceType::GPU>& src, const unsigned int count, const unsigned int dstOffset = 0, const unsigned int srcOffset = 0);
	void assign(const Array<T, DeviceType::CPU>& src, const unsigned int count, const unsigned int dstOffset = 0, const unsigned int srcOffset = 0);
	void assign(const std::vector<T>& src, const unsigned int count, const unsigned int dstOffset = 0, const unsigned int srcOffset = 0);

	friend std::ostream& operator<<(std::ostream &out, const Array<T, DeviceType::GPU>& dArray)
	{
		Array<T, DeviceType::CPU> hArray;
		hArray.assign(dArray);

		out << hArray;

		return out;
	}

private:
	T* mData = nullptr;
	unsigned int mTotalNum = 0;
	unsigned int mBufferNum = 0;
};

template<typename T>
using DArray = Array<T, DeviceType::GPU>;

template<typename T>
void Array<T, DeviceType::GPU>::resize(const unsigned int n)
{
	if (mTotalNum == n) return;

	if (n == 0) {
		clear();
		return;
	}

	int exp = std::ceil(std::log2(float(n)));

	int bound = std::pow(2, exp);

	if (n > mBufferNum || n <= mBufferNum / 2) {
		clear();

		mTotalNum = n; 	
		mBufferNum = bound;

		cudaMalloc(&mData, bound * sizeof(T));
	}
	else
		mTotalNum = n;
}

template<typename T>
void Array<T, DeviceType::GPU>::clear()
{
	if (mData != nullptr)
	{
		cudaFree((void*)mData);
	}

	mData = nullptr;
	mTotalNum = 0;
	mBufferNum = 0;
}

template<typename T>
void Array<T, DeviceType::GPU>::reset()
{
	cudaMemset((void*)mData, 0, mTotalNum * sizeof(T));
}

template<typename T>
void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::GPU>& src)
{
	if (mTotalNum != src.size())
		this->resize(src.size());

	cudaMemcpy(mData, src.begin(), src.size() * sizeof(T), cudaMemcpyDeviceToDevice);
}

template<typename T>
void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::CPU>& src)
{
	if (mTotalNum != src.size())
		this->resize(src.size());

	cudaMemcpy(mData, src.begin(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
}


template<typename T>
void Array<T, DeviceType::GPU>::assign(const std::vector<T>& src)
{
	if (mTotalNum != src.size())
		this->resize((unsigned int)src.size());

	cudaMemcpy(mData, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void Array<T, DeviceType::GPU>::assign(const std::vector<T>& src, const unsigned int count, const unsigned int dstOffset, const unsigned int srcOffset)
{
	cudaMemcpy(mData + dstOffset, src.data() + srcOffset, count * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::CPU>& src, const unsigned int count, const unsigned int dstOffset, const unsigned int srcOffset)
{
	cudaMemcpy(mData + dstOffset, src.begin() + srcOffset, count * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::GPU>& src, const unsigned int count, const unsigned int dstOffset, const unsigned int srcOffset)
{
	cudaMemcpy(mData + dstOffset, src.begin() + srcOffset, count * sizeof(T), cudaMemcpyDeviceToDevice);
}