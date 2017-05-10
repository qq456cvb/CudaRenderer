#include "BxDF.cuh"
#include "MonteCarlo.cuh"
#include "math_constants.h"

__host__ __device__ inline float cosTheta(const Vector &w) {
  return w.z;
}
__host__ __device__ bool BxDF::match(BxDFType flags) const {
  return (type & flags) == type;
}

__host__ __device__ Vector BxDF::sample_f(const Vector &wo, Vector *wi, float *pdf) const {
  *wi = cosineSampleHemisphere();
  if (wo.z < 0.) wi->z *= -1.f;
  *pdf = this->pdf(wo, *wi);
  return this->f(wo, *wi);
}

__host__ __device__ float BxDF::pdf(const Vector &wo, const Vector &wi) const {
  if (wi.z * wo.z > 0)
  {
    return fabsf(cosTheta(wi)) / CUDART_PI_F;
  }
  else {
    return 0;
  }
}