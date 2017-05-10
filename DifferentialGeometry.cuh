//
// Created by Neil on 01/04/2017.
//
#ifndef CUDARENDERER_DIFFERENTIAL_GEOMETRY_H
#define CUDARENDERER_DIFFERENTIAL_GEOMETRY_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Normal.cuh"
#include "Point.cuh"

__host__ __device__ class Shape;

class DifferentialGeometry {
public:
  Normal n;
  Point p;
  const Shape *shape;
  Vector dpdu, dpdv;

  __host__ __device__ DifferentialGeometry();
  __host__ __device__ DifferentialGeometry(const DifferentialGeometry &dg);
  __host__ __device__ DifferentialGeometry(const Point &pt, const Normal &normal,
    const Shape *sh, const Vector& dppduu, const Vector& dppdvv);
};


#endif