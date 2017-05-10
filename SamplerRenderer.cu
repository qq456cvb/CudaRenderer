#include "SamplerRenderer.cuh"
#include "Intersection.cuh"

__host__ __device__ SamplerRenderer::SamplerRenderer()
{
}

__host__ __device__ SamplerRenderer::SamplerRenderer(SurfaceIntegrator *si) :
  surface_integrator(si) {

}

__host__ __device__ SamplerRenderer::~SamplerRenderer()
{
  if (surface_integrator)
  {
    delete surface_integrator;
  }
}

__host__ __device__ void SamplerRenderer::setSharedNodes(Scene *scene, LinearBVHNode *nodes) {
  surface_integrator->setSharedNodes(scene, nodes);
}

__host__ __device__ Vector SamplerRenderer::Li(const Scene *scene, const Ray &ray) const {
  Vector Li(0);
  Intersection isect;
  Ray r = ray;
  if (scene->intersect(r, &isect))
  {
    Li = surface_integrator->Li(scene, this, r, isect);
  }
  else {
    for (size_t i = 0; i < scene->n_lights; i++)
    {

    }
  }
  return Li;
}