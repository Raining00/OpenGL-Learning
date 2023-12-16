#pragma once
#include <glm/vec4.hpp>
#include <iostream>
#include "VectorBase.h"
/*
	* Vector<T,4> are defined for C++ fundamental integer types and floating-point types
*/
template <typename T>
class Vector<T, 4>
{
public:
	typedef T VarType;

	__device__ __host__ Vector();
	__device__ __host__ explicit Vector(T);
	__device__ __host__ Vector(T x, T y, T z, T w);
	__device__ __host__ Vector(const Vector<T, 4>&);
	__device__ __host__ ~Vector();

	__device__ __host__ static int dims() { return 4; }

	__device__ __host__ T& operator[] (unsigned int);
	__device__ __host__ const T& operator[] (unsigned int) const;

	__device__ __host__ const Vector<T, 4> operator+ (const Vector<T, 4>&) const;
	__device__ __host__ Vector<T, 4>& operator+= (const Vector<T, 4>&);
	__device__ __host__ const Vector<T, 4> operator- (const Vector<T, 4>&) const;
	__device__ __host__ Vector<T, 4>& operator-= (const Vector<T, 4>&);
	__device__ __host__ const Vector<T, 4> operator* (const Vector<T, 4>&) const;
	__device__ __host__ Vector<T, 4>& operator*= (const Vector<T, 4>&);
	__device__ __host__ const Vector<T, 4> operator/ (const Vector<T, 4>&) const;
	__device__ __host__ Vector<T, 4>& operator/= (const Vector<T, 4>&);


	__device__ __host__ Vector<T, 4>& operator= (const Vector<T, 4>&);

	__device__ __host__ bool operator== (const Vector<T, 4>&) const;
	__device__ __host__ bool operator!= (const Vector<T, 4>&) const;

	__device__ __host__ const Vector<T, 4> operator* (T) const;
	__device__ __host__ const Vector<T, 4> operator- (T) const;
	__device__ __host__ const Vector<T, 4> operator+ (T) const;
	__device__ __host__ const Vector<T, 4> operator/ (T) const;

	__device__ __host__ Vector<T, 4>& operator+= (T);
	__device__ __host__ Vector<T, 4>& operator-= (T);
	__device__ __host__ Vector<T, 4>& operator*= (T);
	__device__ __host__ Vector<T, 4>& operator/= (T);

	__device__ __host__ const Vector<T, 4> operator - (void) const;

	__device__ __host__ T norm() const;
	__device__ __host__ T normSquared() const;
	__device__ __host__ Vector<T, 4>& normalize();
	//__device__ __host__ Vector<T, 4> cross(const Vector<T, 4>&) const; 
	__device__ __host__ T dot(const Vector<T, 4>&) const;
	//    __device__ __host__ const SquareMatrix<T,4> outerProduct(const Vector<T,4>&) const;

	__device__ __host__ Vector<T, 4> minimum(const Vector<T, 4>&) const;
	__device__ __host__ Vector<T, 4> maximum(const Vector<T, 4>&) const;

	__device__ __host__ T* getDataPtr() { return &data_.x; }

	friend std::ostream& operator<<(std::ostream& out, const Vector<T, 4>& vec)
	{
		out << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << vec[3] << ")";
		return out;
	}

public:
	union
	{
#ifdef VK_BACKEND
		DYN_ALIGN_16 glm::tvec4<T> data_; //default: zero vector
		struct { T x, y, z, w, dummy; };
#else
		glm::tvec4<T> data_; //default: zero vector
		struct { T x, y, z, w; };
#endif // VK_BACKEND
	};
};

template class Vector<float, 4>;
template class Vector<double, 4>;
//convenient typedefs 
typedef Vector<float, 4>			Vec4f;
typedef Vector<double, 4>			Vec4d;
typedef Vector<int, 4>				Vec4i;
typedef Vector<unsigned int, 4>		Vec4u;

#include <limits>
#include <glm/gtx/norm.hpp>

template <typename T>
__device__ __host__ Vector<T, 4>::Vector()
	:Vector(0) //delegating ctor
{
}

template <typename T>
__device__ __host__ Vector<T, 4>::Vector(T x)
	: Vector(x, x, x, x) //delegating ctor
{
}

template <typename T>
__device__ __host__ Vector<T, 4>::Vector(T x, T y, T z, T w)
	: data_(x, y, z, w)
{
}

template <typename T>
__device__ __host__ Vector<T, 4>::Vector(const Vector<T, 4>& vec)
	: data_(vec.data_)
{

}

template <typename T>
__device__ __host__ Vector<T, 4>::~Vector()
{
}

template <typename T>
__device__ __host__ T& Vector<T, 4>::operator[] (unsigned int idx)
{
	return const_cast<T&> (static_cast<const Vector<T, 4> &>(*this)[idx]);
}

template <typename T>
__device__ __host__ const T& Vector<T, 4>::operator[] (unsigned int idx) const
{
	return data_[idx];
}

template <typename T>
__device__ __host__ const Vector<T, 4> Vector<T, 4>::operator+ (const Vector<T, 4>& vec2) const
{
	return Vector<T, 4>(*this) += vec2;
}

template <typename T>
__device__ __host__ Vector<T, 4>& Vector<T, 4>::operator+= (const Vector<T, 4>& vec2)
{
	data_ += vec2.data_;
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 4> Vector<T, 4>::operator- (const Vector<T, 4>& vec2) const
{
	return Vector<T, 4>(*this) -= vec2;
}

template <typename T>
__device__ __host__ Vector<T, 4>& Vector<T, 4>::operator-= (const Vector<T, 4>& vec2)
{
	data_ -= vec2.data_;
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 4> Vector<T, 4>::operator*(const Vector<T, 4>& vec2) const
{
	return Vector<T, 4>(data_[0] * vec2[0], data_[1] * vec2[1], data_[2] * vec2[2], data_[3] * vec2[3]);
}

template <typename T>
__device__ __host__ Vector<T, 4>& Vector<T, 4>::operator*=(const Vector<T, 4>& vec2)
{
	data_[0] *= vec2.data_[0];	data_[1] *= vec2.data_[1];	data_[2] *= vec2.data_[2]; data_[3] *= vec2.data_[3];
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 4> Vector<T, 4>::operator/(const Vector<T, 4>& vec2) const
{
	return Vector<T, 4>(data_[0] / vec2[0], data_[1] / vec2[1], data_[2] / vec2[2], data_[3] / vec2[3]);
}

template <typename T>
__device__ __host__ Vector<T, 4>& Vector<T, 4>::operator/=(const Vector<T, 4>& vec2)
{
	data_[0] /= vec2.data_[0];	data_[1] /= vec2.data_[1];	data_[2] /= vec2.data_[2]; data_[3] /= vec2.data_[3];
	return *this;
}

template <typename T>
__device__ __host__ Vector<T, 4>& Vector<T, 4>::operator=(const Vector<T, 4>& vec2)
{
	data_ = vec2.data_;
	return *this;
}


template <typename T>
__device__ __host__ bool Vector<T, 4>::operator== (const Vector<T, 4>& vec2) const
{
	return data_ == vec2.data_;
}

template <typename T>
__device__ __host__ bool Vector<T, 4>::operator!= (const Vector<T, 4>& vec2) const
{
	return !((*this) == vec2);
}

template <typename T>
__device__ __host__ const Vector<T, 4> Vector<T, 4>::operator+(T value) const
{
	return Vector<T, 4>(*this) += value;
}

template <typename T>
__device__ __host__ Vector<T, 4>& Vector<T, 4>::operator+= (T value)
{
	data_ += value;
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 4> Vector<T, 4>::operator-(T value) const
{
	return Vector<T, 4>(*this) -= value;
}

template <typename T>
__device__ __host__ Vector<T, 4>& Vector<T, 4>::operator-= (T value)
{
	data_ -= value;
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 4> Vector<T, 4>::operator* (T scale) const
{
	return Vector<T, 4>(*this) *= scale;
}

template <typename T>
Vector<T, 4>& Vector<T, 4>::operator*= (T scale)
{
	data_ *= scale;
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 4> Vector<T, 4>::operator/ (T scale) const
{
	return Vector<T, 4>(*this) /= scale;
}

template <typename T>
__device__ __host__ Vector<T, 4>& Vector<T, 4>::operator/= (T scale)
{
	data_ /= scale;
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 4> Vector<T, 4>::operator-(void) const
{
	Vector<T, 4> res;
	res.data_ = -data_;
	return res;
}

template <typename T>
__device__ __host__ T Vector<T, 4>::norm() const
{
	return glm::length(data_);
}

template <typename T>
__device__ __host__ T Vector<T, 4>::normSquared() const
{
	return glm::length2(data_);
}

template <typename T>
__device__ __host__ Vector<T, 4>& Vector<T, 4>::normalize()
{
	data_ = glm::length(data_) > glm::epsilon<T>() ? glm::normalize(data_) : glm::tvec4<T>(0, 0, 0, 0);
	return *this;
}


template <typename T>
__device__ __host__ T Vector<T, 4>::dot(const Vector<T, 4>& vec2) const
{
	return glm::dot(data_, vec2.data_);
}

template <typename T>
__device__ __host__ Vector<T, 4> Vector<T, 4>::minimum(const Vector<T, 4>& vec2) const
{
	Vector<T, 4> res;
	res[0] = data_[0] < vec2[0] ? data_[0] : vec2[0];
	res[1] = data_[1] < vec2[1] ? data_[1] : vec2[1];
	res[2] = data_[2] < vec2[2] ? data_[2] : vec2[2];
	res[3] = data_[3] < vec2[3] ? data_[3] : vec2[3];
	return res;
}

template <typename T>
__device__ __host__ Vector<T, 4> Vector<T, 4>::maximum(const Vector<T, 4>& vec2) const
{
	Vector<T, 4> res;
	res[0] = data_[0] > vec2[0] ? data_[0] : vec2[0];
	res[1] = data_[1] > vec2[1] ? data_[1] : vec2[1];
	res[2] = data_[2] > vec2[2] ? data_[2] : vec2[2];
	res[3] = data_[3] > vec2[3] ? data_[3] : vec2[3];
	return res;
}

//make * operator commutative
template <typename S, typename T>
__device__ __host__ const Vector<T, 4> operator *(S scale, const Vector<T, 4>& vec)
{
	return vec * (T)scale;
}