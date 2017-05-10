#include "DifferentialGeometry.cuh"

__host__ __device__ DifferentialGeometry::DifferentialGeometry() {
}

__host__ __device__ DifferentialGeometry::DifferentialGeometry(const DifferentialGeometry &dg)
{
  n = dg.n;
  p = dg.p;
  shape = dg.shape;
  dpdu = dg.dpdu;
  dpdv = dg.dpdv;
}

__host__ __device__ DifferentialGeometry::DifferentialGeometry(const Point &pt, const Normal &normal, const Shape *sh,
  const Vector& dppduu, const Vector& dppdvv) :
  p(pt), n(normal), shape(sh), dpdu(dppduu), dpdv(dppdvv) {
}
