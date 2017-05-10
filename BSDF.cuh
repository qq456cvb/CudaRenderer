#ifndef CUDARENDERER_BSDF_H
#define CUDARENDERER_BSDF_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "BxDF.cuh"
#include "Normal.cuh"
#include "DifferentialGeometry.cuh"

class BSDF
{
public:
  DifferentialGeometry dg;
  Normal n;
  int nBxDFs;

  __host__ __device__ BSDF();
  __host__ __device__ BSDF(const DifferentialGeometry &dg_, const Normal &n_);
  __host__ __device__ ~BSDF();
  
  __host__ __device__ Vector worldToLocal(const Vector &v) const;
  __host__ __device__ Vector localToWorld(const Vector &v) const;

  __host__ __device__ int numComponents(const BxDFType &flags) const;
  __host__ __device__ Vector sample_f(const Vector &wo_w, Vector *wi_w, float *pdf, BxDFType flags, BxDFType *sampled_type) const;
  __host__ __device__ Vector f(const Vector &wo_w, const Vector &wi_w, BxDFType flags) const;
  __host__ __device__ void add(BxDF *b);
private:
  Vector tn, sn;

  BxDF *bxdfs[8];
};


#endif // !BSDF
