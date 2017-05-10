//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_Normal_H
#define CUDARENDERER_Normal_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Vector.cuh"

class Normal {
public:
  float x;
  float y;
  float z;
  float padding;

  __host__ __device__ Normal();
  __host__ __device__ Normal(const Vector &v);
  __host__ __device__ Normal(const Normal &n);
  __host__ __device__ Normal(float xx, float yy, float zz);

};



#endif
