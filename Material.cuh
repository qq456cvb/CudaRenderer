//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_MATERIAL_H
#define CUDARENDERER_MATERIAL_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Vector.cuh"
#include "DifferentialGeometry.cuh"
#include "BSDF.cuh"

class Material
{
public:
  __host__ __device__ virtual void getBSDF(BSDF *bsdf, const DifferentialGeometry &dg) const = 0;
};

#endif