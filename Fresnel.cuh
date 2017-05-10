//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_Fresnel_H
#define CUDARENDERER_Fresnel_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Vector.cuh"

__host__ __device__ inline float cosTheta(const Vector &w);

class Fresnel
{
public:
  __host__ __device__ virtual Vector evaluate(float) const = 0;
};

__host__ __device__ class FresnelAllReflect
{
public:
  __host__ __device__ Vector evaluate(float) const {
    return Vector(1.f);
  }
};

class FresnelDielectric : public Fresnel
{
  float eta_i, eta_t;
public:
  __host__ __device__ FresnelDielectric(float ei, float et);

  __host__ __device__ Vector evaluate(float cosi) const override;
};

#endif