#include "PlasticMaterial.cuh"

__host__ __device__ PlasticMaterial::PlasticMaterial() :
  tex_diff(NULL), tex_spec(NULL), diff(0) {

}

__host__ __device__ PlasticMaterial::~PlasticMaterial() {
  if (tex_spec)
  {
    delete tex_spec;
    tex_spec = NULL;
  }
  if (tex_diff)
  {
    delete tex_diff;
    tex_diff = NULL;
  }
}

__host__ __device__ void PlasticMaterial::getBSDF(BSDF *bsdf, const DifferentialGeometry &dg) const {
  //return NULL;
  //return diff * tex_diff->evaluate(dg) + (1.f - diff) * tex_spec->evaluate(dg);
}