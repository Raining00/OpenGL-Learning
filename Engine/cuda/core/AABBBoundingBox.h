#pragma once
#include <device_types.h>
#include "Vector3.h"

class AABB
{
public:
	__device__ __host__ AABB();
	__device__ __host__ AABB(const Vec3f& p0, const Vec3f& p1);
	__device__ __host__ AABB(const AABB& box);

	__device__ __host__ bool intersect(const AABB& abox, AABB& interBox) const;
	__device__ __host__ bool checkOverlap(const AABB& abox) const;

	__device__ __host__ AABB merge(const AABB& aabb) const;

	// __device__ __host__ bool meshInsert(const TTriangle3D<float>& tri) const;
	__device__ __host__ bool isValid();

	// __device__ __host__ TOrientedBox3D<float> rotate(const Matrix3D& mat);

	__device__ __host__ inline float length(unsigned int i) const  { return v1[i] - v0[i]; }
	Vec3f v0;
	Vec3f v1;
};

__device__ __host__ AABB::AABB()
{
	v0 = Vec3f(float(-1));
	v1 = Vec3f(float(1));
}

__device__ __host__ AABB::AABB(const Vec3f& p0, const Vec3f& p1)
{
	v0 = p0;
	v1 = p1;
}

__device__ __host__ AABB::AABB(const AABB& box)
{
	v0 = box.v0;
	v1 = box.v1;
}

__device__ __host__ bool AABB::intersect(const AABB& abox, AABB& interBox) const
{
	for (int i = 0; i < 3; i++)
	{
		if (v1[i] <= abox.v0[i] || v0[i] >= abox.v1[i])
		{
			return false;
		}
	}

	interBox.v0 = v0.maximum(abox.v0);
	interBox.v1 = v1.minimum(abox.v1);

	for (int i = 0; i < 3; i++)
	{
		if (v1[i] <= abox.v1[i])
		{
			interBox.v1[i] = v1[i];
		}
		else
		{
			interBox.v1[i] = abox.v1[i];
		}

		if (v0[i] <= abox.v0[i])
		{
			interBox.v0[i] = abox.v0[i];
		}
		else
		{
			interBox.v0[i] = v0[i];
		}
	}

	return true;
}

__device__ __host__ bool AABB::checkOverlap(const AABB& abox) const
{
	for (int i = 0; i < 3; i++)
	{
		if (v1[i] <= abox.v0[i] || v0[i] >= abox.v1[i])
		{
			return false;
		}
	}

	return true;
}

__device__ __host__ AABB AABB::merge(const AABB& aabb) const
{
	AABB ret;
	ret.v0 = v0.minimum(aabb.v0);
	ret.v1 = v1.maximum(aabb.v1);

	return ret;
}
