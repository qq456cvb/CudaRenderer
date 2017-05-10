//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_PLANE_H
#define CUDARENDERER_PLANE_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Ray.cuh"
#include "Shape.cuh"
#include "Normal.cuh"
#include "DifferentialGeometry.cuh"

class Plane : public Shape {
  Normal n;
  float d;

public:
  __host__ __device__ Plane(Normal normal, float distance);

  __host__ __device__ bool intersect(const Ray &ray, float *t_hit, DifferentialGeometry *dg) const override;
  __host__ __device__ virtual int refine(Shape **&shapes) const override;
  __host__ __device__ BBox worldBound() const override;
};



#endif //CUDARENDERER_PLANE_H
