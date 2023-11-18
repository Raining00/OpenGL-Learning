#pragma once
#include <cuda_runtime.h>
#include "Vector3.h"

template <typename T>
inline __device__ __host__ T clamp(const T& v, const T& lo, const T& hi)
{
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

template <typename T>
inline __device__ __host__  Vector<T,2> clamp(const Vector<T,2>& v, const Vector<T,2>& lo, const Vector<T,2>& hi)
{
    Vector<T,2> ret;
    ret[0] = (v[0] < lo[0]) ? lo[0] : (hi[0] < v[0]) ? hi[0] : v[0];
    ret[1] = (v[1] < lo[1]) ? lo[1] : (hi[1] < v[1]) ? hi[1] : v[1];

    return ret;
}

template <typename T>
inline __device__ __host__  Vector<T, 3> clamp(const Vector<T, 3>& v, const Vector<T, 3>& lo, const Vector<T, 3>& hi)
{
    Vector<T, 3> ret;
    ret[0] = (v[0] < lo[0]) ? lo[0] : (hi[0] < v[0]) ? hi[0] : v[0];
    ret[1] = (v[1] < lo[1]) ? lo[1] : (hi[1] < v[1]) ? hi[1] : v[1];
    ret[2] = (v[2] < lo[2]) ? lo[2] : (hi[2] < v[2]) ? hi[2] : v[2];

    return ret;
}

template <typename T>
inline __device__ __host__  T abs(const T& v)
{
    return v < T(0) ? - v : v;
}

template <typename T>
inline __device__ __host__  Vector<T, 2> abs(const Vector<T, 2>& v)
{
    Vector<T, 2> ret;
    ret[0] = (v[0] < T(0)) ? -v[0] : v[0];
    ret[1] = (v[1] < T(0)) ? -v[1] : v[1];

    return ret;
}

template <typename T>
inline __device__ __host__  Vector<T, 3> abs(const Vector<T, 3>& v)
{
    Vector<T, 3> ret;
    ret[0] = (v[0] < T(0)) ? -v[0] : v[0];
    ret[1] = (v[1] < T(0)) ? -v[1] : v[1];
    ret[2] = (v[2] < T(0)) ? -v[2] : v[2];

    return ret;
}


template <typename T>
inline __device__ __host__  T minimum(const T& v0, const T& v1)
{
    return v0 < v1 ? v0 : v1;
}

template <typename T>
inline __device__ __host__  Vector<T, 2> minimum(const Vector<T, 2>& v0, const Vector<T, 2>& v1)
{
    Vector<T, 2> ret;
    ret[0] = (v0[0] < v1[0]) ? v0[0] : v1[0];
    ret[1] = (v0[1] < v1[1]) ? v0[1] : v1[1];

    return ret;
}

template <typename T>
inline __device__ __host__  Vector<T, 3> minimum(const Vector<T, 3>& v0, const Vector<T, 3>& v1)
{
    Vector<T, 3> ret;
    ret[0] = (v0[0] < v1[0]) ? v0[0] : v1[0];
    ret[1] = (v0[1] < v1[1]) ? v0[1] : v1[1];
    ret[2] = (v0[2] < v1[2]) ? v0[2] : v1[2];

    return ret;
}

template <typename T>
inline __device__ __host__  T maximum(const T& v0, const T& v1)
{
    return v0 > v1 ? v0 : v1;
}

template <typename T>
inline __device__ __host__  Vector<T, 2> maximum(const Vector<T, 2>& v0, const Vector<T, 2>& v1)
{
    Vector<T, 2> ret;
    ret[0] = (v0[0] > v1[0]) ? v0[0] : v1[0];
    ret[1] = (v0[1] > v1[1]) ? v0[1] : v1[1];

    return ret;
}

template <typename T>
inline __device__ __host__  Vector<T, 3> maximum(const Vector<T, 3>& v0, const Vector<T, 3>& v1)
{
    Vector<T, 3> ret;
    ret[0] = (v0[0] > v1[0]) ? v0[0] : v1[0];
    ret[1] = (v0[1] > v1[1]) ? v0[1] : v1[1];
    ret[2] = (v0[2] > v1[2]) ? v0[2] : v1[2];

    return ret;
}