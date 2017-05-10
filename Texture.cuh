//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_TEXTURE_H
#define CUDARENDERER_TEXTURE_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Vector.cuh"
#include "DifferentialGeometry.cuh"

class Texture
{
public:
  __host__ __device__ virtual Vector evaluate(const DifferentialGeometry &dg) const = 0;
};

#endif