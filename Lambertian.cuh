
#ifndef CUDARENDERER_Lambertian_H
#define CUDARENDERER_Lambertian_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Vector.cuh"
#include "BxDF.cuh"


class Lambertian : public BxDF {
public:
  __host__ __device__ Lambertian(const Vector &reflectance);
  __host__ __device__ Vector f(const Vector &wo, const Vector &wi) const;
private:
  Vector R;
};

#endif