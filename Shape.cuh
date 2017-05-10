//
// Created by Neil on 01/04/2017.
//

#ifndef CUDARENDERER_SHAPE_H
#define CUDARENDERER_SHAPE_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Ray.cuh"
#include "Transform.cuh"
#include "DifferentialGeometry.cuh"
#include "BBox.cuh"

class Shape {
public:
  uint32_t shape_id;
  Transform world_to_shape;
  Transform shape_to_world;

  //    static uint32_t next_shape_id;

  __host__ __device__ Shape() :
    shape_id(0) {};

  __host__ __device__ Shape(const Transform &o2w, const Transform &w2o);

  __host__ __device__ ~Shape() {};

  // intersect routines
  __host__ __device__ virtual bool intersect(const Ray &ray, float *t_hit, DifferentialGeometry *dg) const = 0;
  __host__ __device__ virtual int refine(Shape **&shapes) const = 0;
  __host__ __device__ virtual BBox worldBound() const = 0;
};

//__host__ __device__ uint32_t Shape::next_shape_id = 0;

#endif //CUDARENDERER_SHAPE_H
