//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_Vector_H
#define CUDARENDERER_Vector_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
class Normal;

class Vector {
public:
  float x;
  float y;
  float z;
  float padding;

  __host__ __device__ Vector();
  __host__ __device__ Vector(float v);
  __host__ __device__ Vector(const Vector &w);
  __host__ __device__ Vector(float xx, float yy, float zz);

  __host__ __device__ float dot(const Vector &w) const;
  __host__ __device__ float dot(const Normal &n) const;
  __host__ __device__ Vector cross(const Vector &w) const;
  __host__ __device__ Vector cross(const Normal &n) const;

  __host__ __device__ Vector operator+(float s) const;
  __host__ __device__ Vector operator+(const Vector &w) const;
  __host__ __device__ Vector& operator+=(float s);
  __host__ __device__ Vector& operator+=(const Vector &v);

  __host__ __device__ Vector operator-(float s) const;
  __host__ __device__ Vector operator-(const Vector &w) const;
  __host__ __device__ Vector& operator-=(float s);

  __host__ __device__ Vector operator*(float s) const;
  __host__ __device__ Vector operator*(const Vector &w) const;
  __host__ __device__ Vector& operator*=(float s);

  __host__ __device__ Vector operator/(float s) const;
  __host__ __device__ Vector operator/(const Vector &w) const;
  __host__ __device__ Vector& operator/=(float s);

  __host__ __device__ float norm() const;
  __host__ __device__ Vector& normalize();
  __host__ __device__ Vector normalized() const;
};

__host__ __device__ Vector operator-(const Vector &v);
__host__ __device__ Vector operator*(float s, const Vector &v);
__host__ __device__ void coordinateSystem(const Vector &v1, Vector *v2, Vector *v3);

#endif //CUDARENDERER_Vector_H
