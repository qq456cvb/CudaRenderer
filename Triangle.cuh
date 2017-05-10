#ifndef CUDARENDERER_TRIANGLE_H
#define CUDARENDERER_TRIANGL_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Ray.cuh"
#include "DifferentialGeometry.cuh"
#include "Shape.cuh"
#include "TriangleMesh.cuh"

class Triangle : public Shape
{
public:
  __host__ __device__ Triangle();
  __host__ __device__ Triangle(const Transform &o2w, const Transform &w2o,
    TriangleMesh *m, int n);
  __host__ __device__ ~Triangle();

  __host__ __device__ bool intersect(const Ray &ray, float *t_hit, DifferentialGeometry *dg) const override;
  __host__ __device__ virtual int refine(Shape **&triangles)  const override;
  __host__ __device__ BBox worldBound() const override;
  TriangleMesh *mesh;
  int *v;
  int n_idx;
private:
  
};

#endif //CUDARENDERER_SPHERE_H