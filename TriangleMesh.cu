#include "TriangleMesh.cuh"
#include "Triangle.cuh"
#include "CuTexture.cuh"

__host__ __device__ TriangleMesh::TriangleMesh()
{
}

__host__ __device__ TriangleMesh::TriangleMesh(const Transform &o2w, const Transform &w2o,
  int nt, int nv, int *vi, Point *P,
  Normal *N) :
  Shape(o2w, w2o),
  n_tris(nt), n_verts(nv),
  vertex_index(vi), n(N), p(P)
{
  // in world coordinates
  for (int i = 0; i < nv; ++i) {
    p[i] = o2w(P[i]);
  }

  /*t = new Triangle[n_tris];
  for (int i = 0; i < n_tris; ++i) {
    t[i] = Triangle(o2w, w2o, (TriangleMesh *)this, i);
  }*/
}

__host__ __device__ TriangleMesh::~TriangleMesh()
{
}

__host__ __device__ bool TriangleMesh::intersect(const Ray &ray, float *t_hit, DifferentialGeometry *dg) const {
  return false;
  bool isect = false;
  for (int i = 0; i < n_tris; ++i) {
    isect |= t[i].intersect(ray, t_hit, dg);
  }
  return isect;
}

__host__ __device__ int TriangleMesh::refine(Shape **&triangles) const {
  triangles = reinterpret_cast<Shape **>(new Triangle*[n_tris]);
  for (int i = 0; i < n_tris; i++)
  {
    triangles[i] = new Triangle(shape_to_world, world_to_shape, (TriangleMesh *)this, i);
  }
  return n_tris;
}

__host__ __device__ BBox TriangleMesh::worldBound() const {
  BBox box;
  for (int i = 0; i < n_verts; i++) {
    box = unionBox(box, p[i]);
  }
  return box;
}