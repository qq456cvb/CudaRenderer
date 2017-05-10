#include "Intersection.cuh"

__host__ __device__ void Intersection::getBSDF(BSDF *bsdf) const {
  primitive->getBSDF(bsdf, dg, obj_to_world);
}