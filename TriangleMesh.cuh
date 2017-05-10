#ifndef CUDARENDERER_TRIANGLE_MESH_H
#define CUDARENDERER_TRIANGLE_MESH_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "CuTexture.cuh"
#include "Ray.cuh"
#include "DifferentialGeometry.cuh"
#include "Shape.cuh"

class Triangle;

class TriangleMesh : public Shape
{
public:
  __host__ __device__ TriangleMesh();
  __host__ __device__ TriangleMesh(const Transform &o2w, const Transform &w2o,
    int nt, int nv, int *vi, Point *P,
    Normal *N);
  __host__ __device__ ~TriangleMesh();

  __host__ __device__ bool intersect(const Ray &ray, float *t_hit, DifferentialGeometry *dg) const;
  __host__ __device__ virtual int refine(Shape **&triangles)  const override;
  __host__ __device__ BBox worldBound() const override;
  int n_tris, n_verts;
  int *vertex_index;
  Point *p;
  Normal *n;
private:
  Triangle *t;
};



#endif