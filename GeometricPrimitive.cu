#include "GeometricPrimitive.cuh"
#include "Triangle.cuh"

__host__ __device__ GeometricPrimitive::GeometricPrimitive(Shape *sh, Material *m) :
  shape(sh), material(m)
{

}

__host__ __device__ GeometricPrimitive::GeometricPrimitive() {

}
__host__ __device__ bool GeometricPrimitive::intersect(const Ray &r, Intersection *isect) const {
  float t_hit;
  if (!shape->intersect(r, &t_hit, &isect->dg))
  {
    return false;
  }

  isect->primitive = this;
  isect->world_to_obj = shape->world_to_shape;
  isect->obj_to_world = shape->shape_to_world;
  isect->shape_id = shape->shape_id;
  isect->prim_id = prim_id;

  r.t_max = t_hit;
  return true;
}

__host__ __device__ void GeometricPrimitive::getBSDF(BSDF *bsdf, const DifferentialGeometry & dg, const Transform & obj_to_world) const {
  material->getBSDF(bsdf, dg);
}

__host__ __device__ int GeometricPrimitive::refine(Primitive **&prims) const {
  Shape **shapes = nullptr;
  int size = shape->refine(shapes);
  prims = reinterpret_cast<Primitive **>(new GeometricPrimitive*[size]);
  for (int i = 0; i < size; i++) {
    prims[i] = new GeometricPrimitive(shapes[i], material);
  }
  return size;
}

__host__ __device__ BBox GeometricPrimitive::worldBound() const {
  return shape->worldBound();
}