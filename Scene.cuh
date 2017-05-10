#ifndef CUDARENDERER_SCENE_H
#define CUDARENDERER_SCENE_H

#include "Primitive.cuh"
#include "Light.cuh"

class Scene
{
public:
  __host__ __device__ Scene();
  __host__ __device__ ~Scene();

  __host__ __device__ bool intersect(Ray &r, Intersection *in) const;

  int n_lights;
  int n_prims;
  Light *lights;
  Primitive **prims;

private:
  
};


#endif