#ifndef CUDARENDERER_MATTE_MATERIAL_H
#define CUDARENDERER_MATTE_MATERIAL_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include "Material.cuh"
#include "Texture.cuh"

class MatteMaterial : public Material {
public:
  __host__ __device__ MatteMaterial(Texture *kd);
  __host__ __device__ void getBSDF(BSDF *bsdf, const DifferentialGeometry &dg) const override;
private:
  Texture *Kd;
};

#endif