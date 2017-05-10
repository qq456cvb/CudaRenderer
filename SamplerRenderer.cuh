#ifndef CUDARENDERER_SAMPLER_RENDERER_H
#define CUDARENDERER_SAMPLER_RENDERER_H
#include "Scene.cuh"
#include "SurfaceIntegrator.cuh"

struct LinearBVHNode;
class SamplerRenderer
{
public:
  __host__ __device__ SamplerRenderer();
  __host__ __device__ SamplerRenderer(SurfaceIntegrator *si);
  __host__ __device__ ~SamplerRenderer();

  __host__ __device__ Vector Li(const Scene *scene, const Ray &ray) const;
  __host__ __device__ void setSharedNodes(Scene *scene, LinearBVHNode *nodes);
private:
  SurfaceIntegrator *surface_integrator;
};



#endif