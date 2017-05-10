#include "Plane.cuh"

__host__ __device__ Plane::Plane(Normal normal, float distance) :
  n(normal), d(distance) {
}

__host__ __device__ bool Plane::intersect(const Ray &r, float *t_hit, DifferentialGeometry *dg) const {
  Ray ray;
  ray = (world_to_shape)(r);
  Vector n_vec = Vector(n.x, n.y, n.z);
  float t_test = -(Vector(ray.o.x, ray.o.y, ray.o.z).dot(n_vec) + d) / (ray.d.dot(n_vec));
  if (t_test < ray.t_min || t_test > ray.t_max)
  {
    return false;
  }
  *t_hit = t_test;
  Point p = ray(t_test);
  const Transform &o2w = shape_to_world;
  Vector dpdu_, dpdv_;
  coordinateSystem(Vector(n.x, n.y, n.z), &dpdu_, &dpdv_);
  *dg = DifferentialGeometry(o2w(p), o2w(n),
    this, o2w(dpdu_), o2w(dpdv_));
  return true;
}

__host__ __device__ int Plane::refine(Shape **&shapes) const {
  return 0;
}

__host__ __device__ BBox Plane::worldBound() const {
  printf("Plane world bound not implemented!\n");
  return BBox();
}