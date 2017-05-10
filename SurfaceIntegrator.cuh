#ifndef CUDARENDERER_SURFACE_INTEGRATOR_H
#define CUDARENDERER_SURFACE_INTEGRATOR_H

#include "Integrator.cuh"
#include "Scene.cuh"

class SamplerRenderer;
struct LinearBVHNode;

class SurfaceIntegrator : public Integrator
{
public:
  __host__ __device__ SurfaceIntegrator();
  __host__ __device__ ~SurfaceIntegrator();

  __host__ __device__ virtual Vector Li(const Scene *scene, const SamplerRenderer *renderer,
    const Ray &ray, const Intersection &isect) const = 0;
  __host__ __device__ virtual void setSharedNodes(Scene *scene, LinearBVHNode *nodes) = 0;
private:

};

#endif