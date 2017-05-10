//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_PLASTIC_MATERIAL_H
#define CUDARENDERER_PLASTIC_MATERIAL_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Vector.cuh"
#include "ConstantTexture.cuh"
#include "Material.cuh"

class PlasticMaterial : public Material
{
public:
  Texture *tex_diff;
  Texture *tex_spec;
  float diff;

  __host__ __device__ PlasticMaterial();
  __host__ __device__ ~PlasticMaterial();

  __host__ __device__ void getBSDF(BSDF *bsdf, const DifferentialGeometry &dg) const override;
};


#endif