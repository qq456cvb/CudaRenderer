#include "DirectLightingIntegrator.cuh"
#include "Intersection.cuh"

__host__ __device__ DirectLightingIntegrator::DirectLightingIntegrator() {

}

__host__ __device__ DirectLightingIntegrator::DirectLightingIntegrator(int max_depth_) :
  max_depth(max_depth_)
{

}

__host__ __device__ DirectLightingIntegrator::~DirectLightingIntegrator() {

}

__host__ __device__ void DirectLightingIntegrator::setSharedNodes(Scene *scene, LinearBVHNode *nodes) {
  scene->prims[0]->setSharedNodes(nodes);
}

__host__ __device__ Vector DirectLightingIntegrator::Li(const Scene *scene, const SamplerRenderer *renderer,
  const Ray &ray, const Intersection &isect) const {
  Vector L(0);
  BSDF bsdf;
  isect.getBSDF(&bsdf);

  Vector wo = -ray.d;
  const Point &p = bsdf.dg.p;
  //printf("P:%f, %f, %f\n", bsdf->dg.p.x, bsdf->dg.p.y, bsdf->dg.p.z);
  const Normal &n = bsdf.dg.n;
  // TODO: add emited area light

  // Estimate Direct
  if (scene->n_lights > 0) {
    L += uniformSampleAllLights(scene, renderer, p, n, wo, &bsdf);
    //L += Vector(fabsf(isect.dg.n.x), isect.dg.n.y, isect.dg.n.z);
    //printf("L:%f, %f, %f\n", L.x, L.y, L.z);
  }

  /*if (ray.depth + 1 < max_depth)
  {
    Ray r(ray);
    r.depth += 1;
    L += specularReflect(r, bsdf, isect, renderer, scene);
  }*/
  //printf("L:%f, %f, %f\n", L.x, L.y, L.z);
  return L;
}