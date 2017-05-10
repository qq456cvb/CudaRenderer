#include "Shape.cuh"

__host__ __device__ Shape::Shape(const Transform &o2w, const Transform &w2o) :
  shape_id(0), shape_to_world(o2w), world_to_shape(w2o)
{

}

