//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_MATRIX4X4_H
#define CUDARENDERER_MATRIX4X4_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <iostream>
#include "Vector.cuh"

class Matrix4x4 {
public:
  float m[4][4];

  __host__ __device__ Matrix4x4() {};
  __host__ __device__ Matrix4x4(float a[4][4]);
  __host__ __device__ Matrix4x4(const Matrix4x4 &mat);
  __host__ __device__ Matrix4x4(float t00, float t01, float t02, float t03,
    float t10, float t11, float t12, float t13,
    float t20, float t21, float t22, float t23,
    float t30, float t31, float t32, float t33);

  __host__ __device__ Matrix4x4 transpose() const;
  __host__ __device__ Matrix4x4 inverse() const;
  __host__ __device__ Matrix4x4 operator*(const Matrix4x4 &mat) const;
  __host__ __device__ static Matrix4x4 identity();
};


#endif
