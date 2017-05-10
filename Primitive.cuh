#ifndef CUDARENDERER_Primitive_H
#define CUDARENDERER_Primitive_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Ray.cuh"
#include "BSDF.cuh"
#include "BBox.cuh"
#include "DifferentialGeometry.cuh"
#include "Transform.cuh"

struct LinearBVHNode;
class Intersection;

class Primitive
{
public:
  __host__ __device__ Primitive();
  __host__ __device__ ~Primitive();

  uint32_t prim_id;

  __host__ __device__ virtual bool intersect(const Ray &r, Intersection *in) const = 0;
  __host__ __device__ virtual void setSharedNodes(LinearBVHNode *nodes) {};
  __host__ __device__ virtual void getBSDF(BSDF *bsdf, const DifferentialGeometry &dg,
    const Transform &obj_to_world) const = 0;
  __host__ __device__ virtual int refine(Primitive **&prims) const = 0;
  __host__ __device__ virtual BBox worldBound() const = 0;
};

#endif