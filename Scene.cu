#include "Scene.cuh"


__host__ __device__ Scene::Scene()
{
}

__host__ __device__ Scene::~Scene()
{
}

__host__ __device__ bool Scene::intersect(Ray &r, Intersection *in) const {
  bool sect = false;
  for (int i = 0; i < n_prims; i++)
  {
    sect |= prims[i]->intersect(r, in);
  }
  return sect;
}