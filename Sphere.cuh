//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_SPHERE_H
#define CUDARENDERER_SPHERE_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Ray.cuh"
#include "DifferentialGeometry.cuh"
#include "Shape.cuh"

class Sphere : public Shape {
public:
  float radius;
  float phi_max;
  float z_min, z_max;
  float theta_min, theta_max;

  __host__ __device__ Sphere(float rad, float z0, float z1, float pm);

  __host__ __device__ bool intersect(const Ray &ray, float *t_hit, DifferentialGeometry *dg) const override;
  __host__ __device__ virtual int refine(Shape **&shapes) const override;
  __host__ __device__ BBox worldBound() const override;
};



#endif //CUDARENDERER_SPHERE_H
