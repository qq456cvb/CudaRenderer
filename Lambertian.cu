#define _USE_MATH_DEFINES
#include "Lambertian.cuh"
#include <cmath>
#include "math_constants.h"
#include <stdio.h>

__host__ __device__ Lambertian::Lambertian(const Vector &reflectance) :
  BxDF(BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE)), R(reflectance)
{
}

__host__ __device__ Vector Lambertian::f(const Vector &wo, const Vector &wi) const {
  //printf("R / PI:%f/%f\n", (R / CUDART_PI_F).x, CUDART_PI_F);
  return R / CUDART_PI_F;
}