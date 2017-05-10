//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_SPECULAR_H
#define CUDARENDERER_SPECULAR_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Vector.cuh"
#include "Fresnel.cuh"
#include "BxDF.cuh"

class SpecularReflection : public BxDF
{
  Vector reflect;
  Fresnel *fresnel;
public:
  __host__ __device__ SpecularReflection(const Vector &r, Fresnel *f);
  __host__ __device__ ~SpecularReflection();

  __host__ __device__ Vector sample_f(const Vector &wo, Vector *wi, float *pdf) const override;
  __host__ __device__ Vector f(const Vector &wo, const Vector &wi) const override;
};



/**************************************************************/

class SpecularTransmission : public BxDF
{
  // refraction index where the ray comes from
  float eta_i;
  // refraction index where the ray injects into
  float eta_t;

  Vector trans;
  FresnelDielectric fresnel;
public:
  __host__ __device__ SpecularTransmission(const Vector &t, float ei, float et);
  __host__ __device__ ~SpecularTransmission();

  __host__ __device__ Vector sample_f(const Vector &wo, Vector *wi, float *pdf) const override;
  __host__ __device__ Vector f(const Vector &wo, const Vector &wi) const override;
};



#endif