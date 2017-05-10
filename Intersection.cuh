#ifndef CUDARENDERER_Intersection_H
#define CUDARENDERER_Intersection_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Ray.cuh"
#include "DifferentialGeometry.cuh"
#include "Primitive.cuh"
#include "Transform.cuh"

class Intersection
{
public:
  DifferentialGeometry dg;
  const Primitive *primitive;
  Transform world_to_obj, obj_to_world;
  uint32_t shape_id, prim_id;

  __host__ __device__  void Intersection::getBSDF(BSDF *bsdf) const;
};

#endif