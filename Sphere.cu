#include "Sphere.cuh"
#include <assert.h>

__host__ __device__ Sphere::Sphere(float rad, float z0, float z1, float pm) :
  radius(rad), z_min(z0), z_max(z1), phi_max(pm) {
  z_min = fmaxf(-rad, z_min);
  z_max = fmaxf(rad, z_max);
  theta_min = acosf(z_max / radius);
  theta_max = acosf(z_min / radius);
  //assert(theta_max > theta_min);
}

__host__ __device__ bool Sphere::intersect(const Ray &r, float *t_hit, DifferentialGeometry *dg) const {
  Ray ray;
  ray = (world_to_shape)(r);
  float A = ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z;
  float B = 2 * (ray.d.x * ray.o.x + ray.d.y * ray.o.y + ray.d.z * ray.o.z);
  float C = ray.o.x * ray.o.x + ray.o.y * ray.o.y + ray.o.z * ray.o.z - radius * radius;

  float delta = B * B - 4 * A * C;
  if (delta <= 0) return false;

  float delta_root = sqrtf(delta);
  float t1, t2, t_test;
  t1 = (-B - delta_root) / (A * 2);
  t2 = (-B + delta_root) / (A * 2);

  if (t1 > ray.t_max || t2 < ray.t_min) return false;
  t_test = t1;
  if (t1 < ray.t_min) t_test = t2;
  if (t_test > ray.t_max) return false;

  *t_hit = t_test;
  Point p = ray(t_test);
  const Transform &o2w = shape_to_world;

  // dg
  float theta = acosf(p.z / radius);
  float inv_rad = 1.f / sqrtf(p.x * p.x + p.y * p.y);
  float cos_phi = p.x * inv_rad;
  float sin_phi = p.y * inv_rad;
  Vector dpdu_ = Vector(-phi_max*p.y, phi_max*p.x, 0);
  Vector dpdv_ = (theta_max-theta_min) * Vector(p.z * cos_phi, p.z * sin_phi, -radius * sinf(theta));
  *dg = DifferentialGeometry(o2w(p), o2w(Vector(p.x, p.y, p.z)).normalized(),
    this, o2w(dpdu_), o2w(dpdv_));
  return true;
}

__host__ __device__ int Sphere::refine(Shape **&shapes) const {
  shapes = (Shape **)new Sphere*[1];
  shapes[0] = new Sphere(this->radius, this->z_min, this->z_max, this->phi_max);
  ((Sphere*)shapes[0])->world_to_shape = this->world_to_shape;
  ((Sphere*)shapes[0])->shape_to_world = this->shape_to_world;
  return 1;
}

__host__ __device__ BBox Sphere::worldBound() const {
  printf("Sphere world bound not implemented!\n");
  return BBox();
}