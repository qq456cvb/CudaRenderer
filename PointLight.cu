#include "PointLight.cuh"

__host__ __device__ PointLight::PointLight() {

}

__host__ __device__ PointLight::PointLight(const Transform &l2w_, const Vector intensity_):
  Light(l2w_), intensity(intensity_)
{
  light_pos = light_to_world(Point(0, 0, 0));
}

__host__ __device__ PointLight::~PointLight()
{
}

__host__ __device__ Vector PointLight::sample_L(const Point &p, Vector *wi, float *pdf) const {
  *wi = (light_pos - p).normalized();
  *pdf = 1.f;
  return intensity / distanceSquared(light_pos, p);
}