#include "BSDF.cuh"
#include <stdio.h>

__host__ __device__ BSDF::BSDF() :
  nBxDFs(0), n(0, 0, 0)
{}

__host__ __device__ BSDF::BSDF(const DifferentialGeometry &dg_, const Normal &n_) :
  dg(dg_), n(n_)
{
  sn = dg.dpdu.normalized();
  //printf("sn:%f, %f, %f\n", dg.dpdu.x, dg.dpdu.y, dg.dpdu.z);
  tn = sn.cross(n);
  //printf("tn:%f, %f, %f\n", tn.x, tn.y, tn.z);
}

__host__ __device__ BSDF::~BSDF()
{
  for (int i = 0; i < nBxDFs; i++) {
    if (bxdfs[i]) {
      delete bxdfs[i];
    }
  }
}

__host__ __device__ Vector BSDF::worldToLocal(const Vector &v) const {
  return Vector(v.dot(sn), v.dot(tn), v.dot(n));
}

__host__ __device__ Vector BSDF::localToWorld(const Vector &v) const {
  return Vector(sn.x * v.x + tn.x * v.y + n.x * v.z,
    sn.y * v.x + tn.y * v.y + n.y * v.z,
    sn.z * v.x + tn.z * v.y + n.z * v.z);
}

__host__ __device__ int BSDF::numComponents(const BxDFType &flags) const {
  int match = 0;
  for (int i = 0; i < nBxDFs; i++)
  {
    if (bxdfs[i]->match(flags))
    {
      match++;
    }
  }
  return match;
}

__host__ __device__ Vector BSDF::sample_f(const Vector &wo_w, Vector *wi_w, float *pdf, BxDFType flags, BxDFType *sampled_type) const {
  if (numComponents(flags) == 0)
  {
    *pdf = 0;
    *sampled_type = BxDFType(0);
    return Vector(0);
  }

  BxDF *bxdf = nullptr;
  for (int i = 0; i < nBxDFs; i++) {
    if (bxdfs[i]->match(flags))
    {
      bxdf = bxdfs[i];
      break;
    }
  }

  Vector wo = worldToLocal(wo_w);
  Vector wi;
  *pdf = 0;
  Vector f = bxdf->sample_f(wo, &wi, pdf);
    
  return f;
}

__host__ __device__ Vector BSDF::f(const Vector &wo_w, const Vector &wi_w, BxDFType flags) const {
  Vector wi = worldToLocal(wi_w), wo = worldToLocal(wo_w);
  // same side
  if (wi_w.dot(n) * wo_w.dot(n) > 0) {
    flags = BxDFType(flags & ~BSDF_TRANSMISSION);
  }
  else {
    flags = BxDFType(flags & ~BSDF_REFLECTION);
  }

  Vector f = 0;
  for (int i = 0; i < nBxDFs; i++) {
    if (bxdfs[i]->match(flags))
    {
      f += bxdfs[i]->f(wo, wi);
      //printf("bxdf f: %f\n", f.x);
    }
  }
  return f;
}

__host__ __device__ void BSDF::add(BxDF *b) {
  bxdfs[nBxDFs++] = b;
}