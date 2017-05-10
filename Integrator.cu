#include "Integrator.cuh"
#include "SamplerRenderer.cuh"

__host__ __device__ Integrator::Integrator()
{
}

__host__ __device__ Integrator::~Integrator()
{
}

__host__ __device__ Vector uniformSampleAllLights(const Scene *scene, const SamplerRenderer *renderer, const Point &p,
  const Normal &n, const Vector &wo, BSDF *bsdf) {
  Vector L(0);
  for (size_t i = 0; i < scene->n_lights; i++)
  {
    const Light &light = scene->lights[i];
    L += estimateDirect(scene, renderer, &light, p, n, wo, bsdf, (BxDFType)(BSDF_ALL & ~BSDF_SPECULAR));
  }
  //printf("L:%f %f %f\n", L.x, L.y, L.z);
  return L; 
}

__host__ __device__ Vector estimateDirect(const Scene *scene, const SamplerRenderer *renderer, const Light *light,
  const Point &p, const Normal &n, const Vector &wo, const BSDF *bsdf, BxDFType flags) {
  Vector Ld = 0;
  Vector wi;
  float light_pdf, bsdf_pdf;
  Vector Li = light->sample_L(p, &wi, &light_pdf);
  
  if (Li.norm() > 0 && light_pdf > 0)
  {
    Vector f = bsdf->f(wo, wi, flags);
    Ld = Ld + f * Li * fabsf(wi.dot(n)) / light_pdf;
    /*printf("Li: %f %f %f\n", Li.x, Li.y, Li.z);
    printf("f: %f %f %f\n", f.x, f.y, f.z);
    printf("cos: %f\n", fabsf(wi.dot(n)));*/
  }
  //printf("Ld:%f %f %f\n", Ld.x, Ld.y, Ld.z);
  return Ld;
}

__host__ __device__ Vector specularReflect(const Ray &ray, BSDF *bsdf, const Intersection &isect,
  const SamplerRenderer *renderer, const Scene *scene) {
  Vector wo = -ray.d;
  Vector wi;
  float pdf;
  const Point &p = bsdf->dg.p;
  const Normal &n = bsdf->dg.n;
  BxDFType sampled;
  Vector f = bsdf->sample_f(wo, &wi, &pdf, BxDFType(BSDF_REFLECTION | BSDF_SPECULAR), &sampled);

  Vector L = 0;
  if (pdf > 0 && f.norm() > 0 && wi.dot(Vector(n.x, n.y, n.z)) != 0)
  {
    Vector Li = renderer->Li(scene, ray);
    L = f * Li * wi.dot(n) / pdf;
  }
  return L;
}