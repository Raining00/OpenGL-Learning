#pragma once

#include <glm/vec2.hpp>
#include <iostream>
#include "VectorBase.h"

template<typename T>
class Vector<T, 2>
{
public:
	typedef T VarType;

    __device__ __host__ Vector();
    __device__ __host__ explicit Vector(T);
    __device__ __host__ Vector(T x, T y);
    __device__ __host__ Vector(const Vector<T, 2>&);
    __device__ __host__ ~Vector();

    __device__ __host__ static int dims() { return 2; }

    __device__ __host__ T& operator[] (unsigned int);
    __device__ __host__ const T& operator[] (unsigned int) const;

    __device__ __host__ const Vector<T, 2> operator+ (const Vector<T, 2>&) const;
    __device__ __host__ Vector<T, 2>& operator+= (const Vector<T, 2>&);
    __device__ __host__ const Vector<T, 2> operator- (const Vector<T, 2>&) const;
    __device__ __host__ Vector<T, 2>& operator-= (const Vector<T, 2>&);
    __device__ __host__ const Vector<T, 2> operator* (const Vector<T, 2>&) const;
    __device__ __host__ Vector<T, 2>& operator*= (const Vector<T, 2>&);
    __device__ __host__ const Vector<T, 2> operator/ (const Vector<T, 2>&) const;
    __device__ __host__ Vector<T, 2>& operator/= (const Vector<T, 2>&);


    __device__ __host__ Vector<T, 2>& operator= (const Vector<T, 2>&);

    __device__ __host__ bool operator== (const Vector<T, 2>&) const;
    __device__ __host__ bool operator!= (const Vector<T, 2>&) const;

    __device__ __host__ const Vector<T, 2> operator* (T) const;
    __device__ __host__ const Vector<T, 2> operator- (T) const;
    __device__ __host__ const Vector<T, 2> operator+ (T) const;
    __device__ __host__ const Vector<T, 2> operator/ (T) const;

    __device__ __host__ Vector<T, 2>& operator+= (T);
    __device__ __host__ Vector<T, 2>& operator-= (T);
    __device__ __host__ Vector<T, 2>& operator*= (T);
    __device__ __host__ Vector<T, 2>& operator/= (T);

    __device__ __host__ const Vector<T, 2> operator - (void) const;

    __device__ __host__ T norm() const;
    __device__ __host__ T normSquared() const;
    __device__ __host__ Vector<T, 2>& normalize();
    __device__ __host__ T cross(const Vector<T, 2>&) const;
    __device__ __host__ T dot(const Vector<T, 2>&) const;
    //    __device__ __host__ const SquareMatrix<T,2> outerProduct(const Vector<T,2>&) const;

    __device__ __host__ Vector<T, 2> minimum(const Vector<T, 2>&) const;
    __device__ __host__ Vector<T, 2> maximum(const Vector<T, 2>&) const;

    __device__ __host__ T* getDataPtr() { return &data_.x; }

    friend std::ostream& operator<<(std::ostream& out, const Vector<T, 2>& vec)
    {
        out << "(" << vec[0] << ", " << vec[1] << ", " << ")";
        return out;
    }

public:
    union
    {
#ifdef VK_BACKEND
        DYN_ALIGN_16 glm::tvec2<T> data_; //default: zero vector
        struct { T x, y,dummy; };
#else
        glm::tvec2<T> data_; //default: zero vector
        struct { T x, y; };
#endif // VK_BACKEND
    };
};

template class Vector<float, 2>;
template class Vector<double, 2>;
//convenient typedefs 
typedef Vector<float, 2>			Vec2f;
typedef Vector<double, 2>			Vec2d;
typedef Vector<int, 2>				Vec2i;
typedef Vector<unsigned int, 2>		Vec2u;


#include <limits>
#include <glm/gtx/norm.hpp>

template<typename T>
__device__ __host__ Vector<T, 2>::Vector() :Vector(0)
{
}

template <typename T>
__device__ __host__ Vector<T, 2>::Vector(T x)
    : Vector(x, x) //delegating ctor
{
}

template <typename T>
__device__ __host__ Vector<T, 2>::Vector(T x, T y)
    : data_(x, y)
{
}

template <typename T>
__device__ __host__ Vector<T, 2>::Vector(const Vector<T, 2>& vec)
    : data_(vec.data_)
{
}

template <typename T>
__device__ __host__ Vector<T, 2>::~Vector()
{
}

template <typename T>
__device__ __host__ T& Vector<T, 2>::operator[] (unsigned int idx)
{
    return const_cast<T&> (static_cast<const Vector<T, 2> &>(*this)[idx]);
}

template <typename T>
__device__ __host__ const T& Vector<T, 2>::operator[] (unsigned int idx) const
{
    return data_[idx];
}

template <typename T>
__device__ __host__ const Vector<T, 2> Vector<T, 2>::operator+ (const Vector<T, 2>& vec2) const
{
    return Vector<T, 2>(*this) += vec2;
}

template <typename T>
__device__ __host__ Vector<T, 2>& Vector<T, 2>::operator+= (const Vector<T, 2>& vec2)
{
    data_ += vec2.data_;
    return *this;
}


template <typename T>
__device__ __host__ const Vector<T, 2> Vector<T, 2>::operator- (const Vector<T, 2>& vec2) const
{
    return Vector<T, 2>(*this) -= vec2;
}

template <typename T>
__device__ __host__ Vector<T, 2>& Vector<T, 2>::operator-= (const Vector<T, 2>& vec2)
{
    data_ -= vec2.data_;
    return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 2> Vector<T, 2>::operator*(const Vector<T, 2>& vec2) const
{
    return Vector<T, 2>(data_[0] * vec2[0], data_[1] * vec2[1]);
}


template <typename T>
__device__ __host__ Vector<T, 2>& Vector<T, 2>::operator*=(const Vector<T, 2>& vec2)
{
    data_[0] *= vec2.data_[0];	data_[1] *= vec2.data_[1];
    return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 2> Vector<T, 2>::operator/(const Vector<T, 2>& vec2) const
{
    return Vector<T, 2>(data_[0] / vec2[0], data_[1] / vec2[1]);
}

template <typename T>
__device__ __host__ Vector<T, 2>& Vector<T, 2>::operator/=(const Vector<T, 2>& vec2)
{
    data_[0] /= vec2.data_[0];	data_[1] /= vec2.data_[1];
    return *this;
}

template <typename T>
__device__ __host__ Vector<T, 2>& Vector<T, 2>::operator=(const Vector<T, 2>& vec2)
{
    data_ = vec2.data_;
    return *this;
}


template <typename T>
__device__ __host__ bool Vector<T, 2>::operator== (const Vector<T, 2>& vec2) const
{
    return data_ == vec2.data_;
}


template <typename T>
__device__ __host__ bool Vector<T, 2>::operator!= (const Vector<T, 2>& vec2) const
{
    return !((*this) == vec2);
}

template <typename T>
__device__ __host__ const Vector<T, 2> Vector<T, 2>::operator+(T value) const
{
    return Vector<T, 2>(*this) += value;
}

template <typename T>
__device__ __host__ Vector<T, 2>& Vector<T, 2>::operator+= (T value)
{
    data_ += value;
    return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 2> Vector<T, 2>::operator-(T value) const
{
    return Vector<T, 2>(*this) -= value;
}

template <typename T>
__device__ __host__ Vector<T, 2>& Vector<T, 2>::operator-= (T value)
{
    data_ -= value;
    return *this;
}


template <typename T>
__device__ __host__ const Vector<T, 2> Vector<T, 2>::operator* (T scale) const
{
    return Vector<T, 2>(*this) *= scale;
}

template <typename T>
Vector<T, 2>& Vector<T, 2>::operator*= (T scale)
{
    data_ *= scale;
    return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 2> Vector<T, 2>::operator/ (T scale) const
{
    return Vector<T, 2>(*this) /= scale;
}

template <typename T>
__device__ __host__ Vector<T, 2>& Vector<T, 2>::operator/= (T scale)
{
    data_ /= scale;
    return *this;
}

template <typename T>
__device__ __host__ const Vector<T, 2> Vector<T, 2>::operator-(void) const
{
    Vector<T, 2> res;
    res.data_ = -data_;
    return res;
}


template <typename T>
__device__ __host__ T Vector<T, 2>::norm() const
{
    return glm::length(data_);
}

template <typename T>
__device__ __host__ T Vector<T, 2>::normSquared() const
{
    return glm::length2(data_);
}

template <typename T>
__device__ __host__ Vector<T, 2>& Vector<T, 2>::normalize()
{
    data_ = glm::length(data_) > glm::epsilon<T>() ? glm::normalize(data_) : glm::tvec2<T>(0, 0);
    return *this;
}


template <typename T>
__device__ __host__ T Vector<T, 2>::cross(const Vector<T, 2>& vec2) const
{
    T res;
    // 2D vector only have fake cross product which is a scalar.
    res = data_[0] * vec2[1] - data_[1] * vec2[0];
    return res;
}

template <typename T>
__device__ __host__ T Vector<T, 2>::dot(const Vector<T, 2>& vec2) const
{
    return glm::dot(data_, vec2.data_);
}

template <typename T>
__device__ __host__ Vector<T, 2> Vector<T, 2>::minimum(const Vector<T, 2>& vec2) const
{
    Vector<T, 2> res;
    res[0] = data_[0] < vec2[0] ? data_[0] : vec2[0];
    res[1] = data_[1] < vec2[1] ? data_[1] : vec2[1];
    return res;
}

template <typename T>
__device__ __host__ Vector<T, 2> Vector<T, 2>::maximum(const Vector<T, 2>& vec2) const
{
    Vector<T, 2> res;
    res[0] = data_[0] > vec2[0] ? data_[0] : vec2[0];
    res[1] = data_[1] > vec2[1] ? data_[1] : vec2[1];
    return res;
}

//make * operator commutative
template <typename S, typename T>
__device__ __host__ const Vector<T, 2> operator *(S scale, const Vector<T, 2>& vec)
{
    return vec * (T)scale;
}