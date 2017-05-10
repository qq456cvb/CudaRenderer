#ifndef CUDARENDERER_LIGHT_H
#define CUDARENDERER_LIGHT_H
#include "Transform.cuh"

class Light
{
public:
  __host__ __device__ Light();
  __host__ __device__ Light(const Transform &l2w);
  __host__ __device__ ~Light();

  __host__ __device__ virtual Vector sample_L(const Point &p, Vector *wi, float *pdf) const = 0;
  //virtual Vector Le(const Ray &ray) const = 0;

protected:
  const Transform world_to_light, light_to_world;

};



#endif