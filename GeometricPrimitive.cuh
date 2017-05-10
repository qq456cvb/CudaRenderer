#ifndef CUDARENDERER_GeometricPrimitive_H
#define CUDARENDERER_GeometricPrimitive_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Ray.cuh"
#include "Primitive.cuh"
#include "Intersection.cuh"
#include "Shape.cuh"
#include "Material.cuh"

class GeometricPrimitive : public Primitive
{
  Material *material;
  Shape *shape;
public:
  __host__ __device__ GeometricPrimitive();
  __host__ __device__ GeometricPrimitive(Shape *sh, Material *m);

  __host__ __device__ bool intersect(const Ray &r, Intersection *isect) const override;
  __host__ __device__ void getBSDF(BSDF *bsdf, const DifferentialGeometry & dg, const Transform & obj_to_world) const override;
  __host__ __device__ virtual int refine(Primitive **&prims) const override;
  __host__ __device__ virtual BBox worldBound() const override;
};

#endif