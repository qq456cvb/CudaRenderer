#ifndef CUDARENDERER_MONTE_CARLO_H
#define CUDARENDERER_MONTE_CARLO_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include "Vector.cuh"
#include<curand.h>
#include<curand_kernel.h>

__host__ __device__ void uniformSampleDisk(float *x, float *y);
__host__ __device__ Vector cosineSampleHemisphere();
#endif