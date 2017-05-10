#include "Triangle.cuh"

extern texture<int4, 1, cudaReadModeElementType> vi_texture;
extern texture<float4, 1, cudaReadModeElementType> p_texture;

__host__ __device__ Triangle::Triangle()
{
}

__host__ __device__ Triangle::Triangle(const Transform &o2w, const Transform &w2o,
  TriangleMesh *m, int n) :
  Shape(o2w, w2o) {
  mesh = m;
  v = &mesh->vertex_index[4 * n];
  n_idx = n;
}

__host__ __device__ Triangle::~Triangle()
{
}

__host__ __device__ bool Triangle::intersect(const Ray &ray, float *t_hit, DifferentialGeometry *dg) const {
//  int4 v_idx;
//#ifdef __CUDA_ARCH__
//  v_idx = tex1Dfetch(vi_texture, n_idx);
//
//  Point p1(tex1Dfetch(p_texture, v_idx.x));
//  Point p2(tex1Dfetch(p_texture, v_idx.y));
//  Point p3(tex1Dfetch(p_texture, v_idx.z));
//#else
  const Point &p1 = mesh->p[v[0]];
  const Point &p2 = mesh->p[v[1]];
  const Point &p3 = mesh->p[v[2]];
//#endif // __CUDA_ARCH__
 /* printf("p1: %f, %f, %f\n", p1.x, p1.y, p1.z);
  printf("p2: %f, %f, %f\n", p2.x, p2.y, p2.z);
  printf("p3: %f, %f, %f\n", p3.x, p3.y, p3.z);*/
  
  Vector e1 = p2 - p1;
  Vector e2 = p3 - p1;
  Vector s1 = ray.d.cross(e2);
  
  float divisor = s1.dot(e1);
  
  if (divisor == 0)
  {
    return false;
  }
  float inv_divisor = 1.f / divisor;

  // Compute first barycentric coordinate
  Vector s = ray.o - p1;
  float b1 = s.dot(s1) * inv_divisor;
  if (b1 < 0. || b1 > 1.)
    return false;

  // Compute second barycentric coordinate
  Vector s2 = s.cross(e1);
  float b2 = ray.d.dot(s2) * inv_divisor;
  if (b2 < 0. || b1 + b2 > 1.)
    return false;

  // Compute _t_ to intersection point
  float t = e2.dot(s2) * inv_divisor;
  //printf("t: %f\n", t);
  if (t < ray.t_min || t > ray.t_max)
    return false;

  // Compute triangle partial derivatives
  Vector dpdu, dpdv;
  Normal n = e2.cross(e1).normalized();
  coordinateSystem(Vector(n.x, n.y, n.z), &dpdu, &dpdv);

  *dg = DifferentialGeometry(ray(t), n, this, dpdu, dpdv);
  *t_hit = t;
  return true;
}

__host__ __device__ int Triangle::refine(Shape **&triangles)  const {
  return 0;
}

__host__ __device__ BBox Triangle::worldBound() const {
  const Point &p1 = mesh->p[v[0]];
  
  const Point &p2 = mesh->p[v[1]];
  const Point &p3 = mesh->p[v[2]];
  
  return unionBox(BBox(p1, p2), p3);
}