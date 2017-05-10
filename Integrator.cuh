#ifndef CUDARENDERER_INTEGRATOR_H
#define CUDARENDERER_INTEGRATOR_H
#include "Scene.cuh"

class SamplerRenderer;
class Integrator
{
public:
  __host__ __device__ Integrator();
  __host__ __device__ ~Integrator();

private:

};

__host__ __device__ Vector uniformSampleAllLights(const Scene *scene, const SamplerRenderer *renderer, const Point &p,
  const Normal &n, const Vector &wo, BSDF *bsdf);

__host__ __device__ Vector estimateDirect(const Scene *scene, const SamplerRenderer *renderer, const Light *light,
  const Point &p, const Normal &n, const Vector &wo, const BSDF *bsdf, BxDFType flags);

__host__ __device__ Vector specularReflect(const Ray &ray, BSDF *bsdf, const Intersection &isect,
  const SamplerRenderer *renderer, const Scene *scene);
#endif