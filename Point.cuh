//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_Point_H
#define CUDARENDERER_Point_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Vector.cuh"

class Point {
public:
  float x;
  float y;
  float z;
  //float padding;

  __host__ __device__ Point();
  __host__ __device__ Point(float4 f);
  __host__ __device__ Point(const Point &p);
  __host__ __device__ Point(float xx, float yy, float zz);

  __host__ __device__ Point operator+(const Vector &v) const;
  __host__ __device__ Point& operator+=(const Vector &v);
  __host__ __device__ Point operator-(const Vector &v) const;
  __host__ __device__ Point& operator-=(const Vector &v);

  __host__ __device__ Vector operator-(const Point &p) const;
  __host__ __device__ float operator[](int idx) const;
};

__host__ __device__ float distance(const Point &p1, const Point &p2);
__host__ __device__ float distanceSquared(const Point &p1, const Point &p2);

__host__ __device__ Point operator*(const Point &p, float f);
__host__ __device__ Point operator*(float f, const Point &p);

__host__ __device__ Point operator+(const Point &p1, const Point &p2);
#endif
