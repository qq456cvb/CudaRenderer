#include "BBox.cuh"
#include <float.h>
#include <stdio.h>

__host__ __device__ BBox::BBox()
  : p_min(INFINITY, INFINITY, INFINITY),
  p_max(-INFINITY, -INFINITY, -INFINITY)
{

}

__host__ __device__ BBox::BBox(const Point &p)
  : p_min(p), p_max(p)
{

}

__host__ __device__ BBox::BBox(const Point &p1, const Point &p2) {
  p_min = Point(fminf(p1.x, p2.x), fminf(p1.y, p2.y), fminf(p1.z, p2.z));
  p_max = Point(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y), fmaxf(p1.z, p2.z));
}

__host__ __device__ BBox::~BBox() {

}

__host__ __device__ bool BBox::overlaps(const BBox &b) const {
  bool x = (p_max.x > b.p_min.x) && (p_min.x < b.p_max.x);
  bool y = (p_max.y > b.p_min.y) && (p_min.y < b.p_max.y);
  bool z = (p_max.z > b.p_min.z) && (p_min.z < b.p_max.z);
  return (x && y && z);
}

__host__ __device__ bool BBox::inside(const Point &p) const {
  return (p.x > p_min.x && p.x < p_max.x &&
    p.y > p_min.y && p.y < p_max.y &&
    p.z > p_min.z && p.z < p_max.z);
}

__host__ __device__ void BBox::expand(float delta) {
  p_min -= delta;
  p_max += delta;
}

__host__ __device__ float BBox::surfaceArea() const {
  Vector extent = p_max - p_min;
  return 2.f * (extent.x * extent.y + extent.x * extent.z + extent.y * extent.z);
}

__host__ __device__ float BBox::volumn() const {
  Vector extent = p_max - p_min;
  return extent.x * extent.y * extent.z;
}

__host__ __device__ int BBox::maximumExtent() const {
  Vector diag = p_max - p_min;
  if (diag.x > diag.y && diag.x > diag.z) {
    return 0;
  }
  else if (diag.y > diag.z) {
    return 1;
  }
  else {
    return 2;
  }
}

__host__ __device__ const Point& BBox::operator[](int i) const {
  if (i == 0)
  {
    return p_min;
  }
  else {
    return p_max;
  }
}

__host__ __device__ BBox unionBox(const BBox &b, const Point &p) {
  BBox box;
  box.p_min.x = fminf(b.p_min.x, p.x);
  box.p_min.y = fminf(b.p_min.y, p.y);
  box.p_min.z = fminf(b.p_min.z, p.z);

  box.p_max.x = fmaxf(b.p_max.x, p.x);
  box.p_max.y = fmaxf(b.p_max.y, p.y);
  box.p_max.z = fmaxf(b.p_max.z, p.z);

  return box;
}

__host__ __device__ BBox unionBox(const BBox &b1, const BBox &b2) {
  BBox box;
  box.p_min.x = fminf(b1.p_min.x, b2.p_min.x);
  box.p_min.y = fminf(b1.p_min.y, b2.p_min.y);
  box.p_min.z = fminf(b1.p_min.z, b2.p_min.z);

  box.p_max.x = fmaxf(b1.p_max.x, b2.p_max.x);
  box.p_max.y = fmaxf(b1.p_max.y, b2.p_max.y);
  box.p_max.z = fmaxf(b1.p_max.z, b2.p_max.z);
  return box;
}

