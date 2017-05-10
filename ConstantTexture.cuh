//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_CONSTANT_TEXTURE_H
#define CUDARENDERER_CONSTANT_TEXTURE_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Vector.cuh"
#include "Texture.cuh"
#include "DifferentialGeometry.cuh"

class ConstantTexture : public Texture
{
  Vector color;
public:
  __host__ __device__ ConstantTexture();
  __host__ __device__ ConstantTexture(const Vector &c);
  __host__ __device__ ~ConstantTexture();

  __host__ __device__ Vector evaluate(const DifferentialGeometry &dg) const override;
};

#endif