#include "Light.cuh"

__host__ __device__ Light::Light()
{
}

__host__ __device__ Light::Light(const Transform &l2w) :
  light_to_world(l2w), world_to_light(l2w.inverse())
{
}

__host__ __device__ Light::~Light()
{
}