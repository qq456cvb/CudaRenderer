//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_TRANSFORM_H
#define CUDARENDERER_TRANSFORM_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <iostream>
#include "Matrix4x4.cuh"
#include "Point.cuh"
#include "Ray.cuh"
#include "Normal.cuh"

class Transform {
  Matrix4x4 m, m_inv;

public:
  __host__ __device__ Transform() {};
  __host__ __device__ Transform(const Matrix4x4 &mat);
  __host__ __device__ Transform(const Matrix4x4 &mat, const Matrix4x4 &mat_inv);
  __host__ __device__ Transform inverse() const;
  __host__ __device__ static Transform identity();

  __host__ __device__ static Transform translate(const Vector &delta);
  __host__ __device__ static Transform scale(float x, float y, float z);
  __host__ __device__ static Transform rotateX(float angle);
  __host__ __device__ static Transform rotateY(float angle);
  __host__ __device__ static Transform rotateZ(float angle);

  __host__ __device__ static Transform rotate(float angle, const Vector &axis);
  __host__ __device__ static Transform lookAt(const Point &pos, const Point &target, const Vector &up);

  __host__ __device__ Vector operator()(const Vector &v) const;
  __host__ __device__ Point operator()(const Point &p) const;
  __host__ __device__ Ray operator()(const Ray &ray) const;
  __host__ __device__ Normal operator()(const Normal &n) const;

  __host__ __device__ Transform operator*(const Transform &t) const;
};



#endif
