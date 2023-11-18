#pragma once
#include <glm/vec3.hpp>
#include <iostream>
#include "VectorBase.h"
/*
    * Vector<T,3> are defined for C++ fundamental integer types and floating-point types
*/
template <typename T>
class Vector<T, 3>
{
public:
    typedef T VarType;

    __device__ __host__ Vector();
    __device__ __host__ explicit Vector(T);
    __device__ __host__ Vector(T x, T y, T z);
    __device__ __host__ Vector(const Vector<T, 3>&);
    __device__ __host__ ~Vector();

    __device__ __host__ static int dims() { return 3; }

    __device__ __host__ T& operator[] (unsigned int);
    __device__ __host__ const T& operator[] (unsigned int) const;

    __device__ __host__ const Vector<T, 3> operator+ (const Vector<T, 3> &) const;
    __device__ __host__ Vector<T, 3>& operator+= (const Vector<T, 3> &);
    __device__ __host__ const Vector<T, 3> operator- (const Vector<T, 3> &) const;
    __device__ __host__ Vector<T, 3>& operator-= (const Vector<T, 3> &);
    __device__ __host__ const Vector<T, 3> operator* (const Vector<T, 3> &) const;
    __device__ __host__ Vector<T, 3>& operator*= (const Vector<T, 3> &);
    __device__ __host__ const Vector<T, 3> operator/ (const Vector<T, 3> &) const;
    __device__ __host__ Vector<T, 3>& operator/= (const Vector<T, 3> &);


    __device__ __host__ Vector<T, 3>& operator= (const Vector<T, 3> &);

    __device__ __host__ bool operator== (const Vector<T, 3> &) const;
    __device__ __host__ bool operator!= (const Vector<T, 3> &) const;

    __device__ __host__ const Vector<T, 3> operator* (T) const;
    __device__ __host__ const Vector<T, 3> operator- (T) const;
    __device__ __host__ const Vector<T, 3> operator+ (T) const;
    __device__ __host__ const Vector<T, 3> operator/ (T) const;

    __device__ __host__ Vector<T, 3>& operator+= (T);
    __device__ __host__ Vector<T, 3>& operator-= (T);
    __device__ __host__ Vector<T, 3>& operator*= (T);
    __device__ __host__ Vector<T, 3>& operator/= (T);

    __device__ __host__ const Vector<T, 3> operator - (void) const;

    __device__ __host__ T norm() const;
    __device__ __host__ T normSquared() const;
    __device__ __host__ Vector<T, 3>& normalize();
    __device__ __host__ Vector<T, 3> cross(const Vector<T, 3> &) const;
    __device__ __host__ T dot(const Vector<T, 3>&) const;
    //    __device__ __host__ const SquareMatrix<T,3> outerProduct(const Vector<T,3>&) const;

    __device__ __host__ Vector<T, 3> minimum(const Vector<T, 3>&) const;
    __device__ __host__ Vector<T, 3> maximum(const Vector<T, 3>&) const;

    __device__ __host__ T* getDataPtr() { return &data_.x; }

    friend std::ostream& operator<<(std::ostream &out, const Vector<T, 3>& vec)
    {
        out << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";
        return out;
    }

public:
    union
    {
#ifdef VK_BACKEND
        DYN_ALIGN_16 glm::tvec3<T> data_; //default: zero vector
        struct { T x, y, z, dummy; };
#else
        glm::tvec3<T> data_; //default: zero vector
        struct { T x, y, z; };
#endif // VK_BACKEND
    };
};

template class Vector<float, 3>;
template class Vector<double, 3>;
//convenient typedefs 
typedef Vector<float, 3>	Vec3f;
typedef Vector<double, 3>	Vec3d;
typedef Vector<int, 3>		Vec3i;
typedef Vector<uint, 3>		Vec3u;

#include <limits>
#include <glm/gtx/norm.hpp>

template <typename T>
__device__ __host__ Vector<T, 3>::Vector()
	:Vector(0) //delegating ctor
{
}

template <typename T>
__device__ __host__ Vector<T, 3>::Vector(T x)
	: Vector(x, x, x) //delegating ctor
{
}

template <typename T>
__device__ __host__ Vector<T, 3>::Vector(T x, T y, T z)
	: data_(x, y, z)
{
}

template <typename T>
__device__ __host__ Vector<T, 3>::Vector(const Vector<T, 3>& vec)
	: data_(vec.data_)
{

}

template <typename T>
__device__ __host__ Vector<T, 3>::~Vector()
{
}

template <typename T>
__device__ __host__ T& Vector<T, 3>::operator[] (unsigned int idx)
{
	return const_cast<T &> (static_cast<const Vector<T, 3> &>(*this)[idx]);
}

template <typename T>
__device__ __host__ const T& Vector<T, 3>::operator[] (unsigned int idx) const
{
	return data_[idx];
}

template <typename T>
__device__ __host__ const Vector<T, 3> Vector<T, 3>::operator+ (const Vector<T, 3> &vec2) const
{
	return Vector<T, 3>(*this) += vec2;
}

template <typename T>
__device__ __host__ Vector<T, 3>& Vector<T, 3>::operator+= (const Vector<T, 3> &vec2)
{
	data_ += vec2.data_;
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 3> Vector<T, 3>::operator- (const Vector<T, 3> &vec2) const
{
	return Vector<T, 3>(*this) -= vec2;
}

template <typename T>
__device__ __host__ Vector<T, 3>& Vector<T, 3>::operator-= (const Vector<T, 3> &vec2)
{
	data_ -= vec2.data_;
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 3> Vector<T, 3>::operator*(const Vector<T, 3> &vec2) const
{
	return Vector<T, 3>(data_[0] * vec2[0], data_[1] * vec2[1], data_[2] * vec2[2]);
}

template <typename T>
__device__ __host__ Vector<T, 3>& Vector<T, 3>::operator*=(const Vector<T, 3> &vec2)
{
	data_[0] *= vec2.data_[0];	data_[1] *= vec2.data_[1];	data_[2] *= vec2.data_[2];
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 3> Vector<T, 3>::operator/(const Vector<T, 3> &vec2) const
{
	return Vector<T, 3>(data_[0] / vec2[0], data_[1] / vec2[1], data_[2] / vec2[2]);
}

template <typename T>
__device__ __host__ Vector<T, 3>& Vector<T, 3>::operator/=(const Vector<T, 3> &vec2)
{
	data_[0] /= vec2.data_[0];	data_[1] /= vec2.data_[1];	data_[2] /= vec2.data_[2];
	return *this;
}

template <typename T>
__device__ __host__ Vector<T, 3>& Vector<T, 3>::operator=(const Vector<T, 3> &vec2)
{
	data_ = vec2.data_;
	return *this;
}


template <typename T>
__device__ __host__ bool Vector<T, 3>::operator== (const Vector<T, 3> &vec2) const
{
	return data_ == vec2.data_;
}

template <typename T>
__device__ __host__ bool Vector<T, 3>::operator!= (const Vector<T, 3> &vec2) const
{
	return !((*this) == vec2);
}

template <typename T>
__device__ __host__ const Vector<T, 3> Vector<T, 3>::operator+(T value) const
{
	return Vector<T, 3>(*this) += value;
}

template <typename T>
__device__ __host__ Vector<T, 3>& Vector<T, 3>::operator+= (T value)
{
	data_ += value;
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 3> Vector<T, 3>::operator-(T value) const
{
	return Vector<T, 3>(*this) -= value;
}

template <typename T>
__device__ __host__ Vector<T, 3>& Vector<T, 3>::operator-= (T value)
{
	data_ -= value;
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 3> Vector<T, 3>::operator* (T scale) const
{
	return Vector<T, 3>(*this) *= scale;
}

template <typename T>
Vector<T, 3>& Vector<T, 3>::operator*= (T scale)
{
	data_ *= scale;
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 3> Vector<T, 3>::operator/ (T scale) const
{
	return Vector<T, 3>(*this) /= scale;
}

template <typename T>
__device__ __host__ Vector<T, 3>& Vector<T, 3>::operator/= (T scale)
{
	data_ /= scale;
	return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 3> Vector<T, 3>::operator-(void) const
{
	Vector<T, 3> res;
	res.data_ = -data_;
	return res;
}

template <typename T>
__device__ __host__ T Vector<T, 3>::norm() const
{
	return glm::length(data_);
}

template <typename T>
__device__ __host__ T Vector<T, 3>::normSquared() const
{
	return glm::length2(data_);
}

template <typename T>
__device__ __host__ Vector<T, 3>& Vector<T, 3>::normalize()
{
	data_ = glm::length(data_) > glm::epsilon<T>() ? glm::normalize(data_) : glm::tvec3<T>(0, 0, 0);
	return *this;
}

template <typename T>
__device__ __host__ Vector<T, 3> Vector<T, 3>::cross(const Vector<T, 3>& vec2) const
{
	Vector<T, 3> res;
	res.data_ = glm::cross(data_, vec2.data_);
	return res;
}

template <typename T>
__device__ __host__ T Vector<T, 3>::dot(const Vector<T, 3>& vec2) const
{
	return glm::dot(data_, vec2.data_);
}

template <typename T>
__device__ __host__ Vector<T, 3> Vector<T, 3>::minimum(const Vector<T, 3>& vec2) const
{
	Vector<T, 3> res;
	res[0] = data_[0] < vec2[0] ? data_[0] : vec2[0];
	res[1] = data_[1] < vec2[1] ? data_[1] : vec2[1];
	res[2] = data_[2] < vec2[2] ? data_[2] : vec2[2];
	return res;
}

template <typename T>
__device__ __host__ Vector<T, 3> Vector<T, 3>::maximum(const Vector<T, 3>& vec2) const
{
	Vector<T, 3> res;
	res[0] = data_[0] > vec2[0] ? data_[0] : vec2[0];
	res[1] = data_[1] > vec2[1] ? data_[1] : vec2[1];
	res[2] = data_[2] > vec2[2] ? data_[2] : vec2[2];
	return res;
}

//make * operator commutative
template <typename S, typename T>
__device__ __host__ const Vector<T, 3> operator *(S scale, const Vector<T, 3> &vec)
{
	return vec * (T)scale;
}