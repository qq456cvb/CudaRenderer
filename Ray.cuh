//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_RAY_H
#define CUDARENDERER_RAY_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <iostream>
#include "Vector.cuh"
#include "Point.cuh"
#include <climits>

class Ray {
public:
  Point o;
  Vector d;
  mutable float t_min, t_max;
  int depth;

  __host__ __device__ Ray();
  __host__ __device__ Ray(const Point &origin, const Vector &dir,
    float t_start, float t_end = INFINITY, int d = 0);

  __host__ __device__ Point operator()(float t) const;
};

#endif //CUDARENDERER_RAY_H
