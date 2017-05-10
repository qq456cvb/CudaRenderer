#ifndef CUDARENDERER_DIRECT_LIGHTING_INTEGRATOR_H
#define CUDARENDERER_DIRECT_LIGHTING_INTEGRATOR_H

#include "SurfaceIntegrator.cuh"

class DirectLightingIntegrator : public SurfaceIntegrator
{
public:
  __host__ __device__ DirectLightingIntegrator();
  __host__ __device__ DirectLightingIntegrator(int max_depth_);
  __host__ __device__ ~DirectLightingIntegrator();

  __host__ __device__ Vector Li(const Scene *scene, const SamplerRenderer *renderer,
    const Ray &ray, const Intersection &isect) const override;
  __host__ __device__ virtual void setSharedNodes(Scene *scene, LinearBVHNode *nodes);

private:
  int max_depth;
};


#endif