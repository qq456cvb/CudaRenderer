#include "MonteCarlo.cuh"
#include "math_constants.h"

__host__ __device__ void uniformSampleDisk(float *x, float *y) {
  float u1, u2;
#ifdef  __CUDA_ARCH__
  curandState stat;
  curand_init(0, 0, 0, &stat);
  u1 = curand_uniform(&stat);
  u2 = curand_uniform(&stat);
#else
  srand(NULL);
  u1 = ((float)rand()) / RAND_MAX;
  u2 = ((float)rand()) / RAND_MAX;
#endif

  float r = sqrtf(u1);
  float theta = 2.0f * CUDART_PI_F * u2;
  *x = r * cosf(theta);
  *y = r * sinf(theta);
}

__host__ __device__ Vector cosineSampleHemisphere() {
  Vector ret;
  uniformSampleDisk(&ret.x, &ret.y);
  ret.z = sqrtf(fmaxf(0.f, 1.f - ret.x*ret.x - ret.y*ret.y));
  return ret;
}
