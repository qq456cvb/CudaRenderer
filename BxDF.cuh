//
// Created by Neil on 01/04/2017.
//
#define _USE_MATH_DEFINES
#ifndef CUDARENDERER_BxDF_H
#define CUDARENDERER_BxDF_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Vector.cuh"

enum BxDFType {
  BSDF_REFLECTION = 1,
  BSDF_TRANSMISSION = 1 << 1,
  BSDF_DIFFUSE = 1 << 2,
  BSDF_GLOSSY = 1 << 3,
  BSDF_SPECULAR = 1 << 4,
  BSDF_ALL_TYPES = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR,
  BSDF_ALL_REFLECTION = BSDF_REFLECTION | BSDF_ALL_TYPES,
  BSDF_ALL_TRANSMISSION = BSDF_TRANSMISSION | BSDF_ALL_TYPES,
  BSDF_ALL = BSDF_ALL_REFLECTION | BSDF_ALL_TRANSMISSION
};

class BxDF {
public:
  const BxDFType type;

  __host__ __device__ BxDF(BxDFType t) : type(t) {}
  __host__ __device__ bool match(BxDFType flags) const;
  __host__ __device__ virtual Vector sample_f(const Vector &wo, Vector *wi, float *pdf) const;
  __host__ __device__ virtual float pdf(const Vector &wo, const Vector &wi) const;
  __host__ __device__ virtual Vector f(const Vector &wo,const Vector &wi) const = 0;
};

#endif