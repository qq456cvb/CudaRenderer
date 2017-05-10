#include "MatteMaterial.cuh"
#include "Lambertian.cuh"
#include <stdio.h>

__host__ __device__ MatteMaterial::MatteMaterial(Texture *kd) :
  Kd(kd) {

}

__host__ __device__ void MatteMaterial::getBSDF(BSDF *bsdf,const DifferentialGeometry &dg) const {
  bsdf->dg = dg;
  bsdf->n = dg.n;
  //printf("P:%f, %f, %f\n", bsdf->dg.p.x, bsdf->dg.p.y, bsdf->dg.p.z);
  Vector r = Kd->evaluate(dg);
  bsdf->add(new Lambertian(r));
}