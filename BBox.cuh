#ifndef CUDARENDERER_BBOX_H
#define CUDARENDERER_BBOX_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Point.cuh"

class BBox {
public:
  __host__ __device__ BBox();
  __host__ __device__ BBox(const Point &p);
  __host__ __device__ BBox(const Point &p1, const Point &p2);
  __host__ __device__ ~BBox();

  __host__ __device__ bool overlaps(const BBox &b) const;
  __host__ __device__ bool inside(const Point &p) const;
  __host__ __device__ void expand(float delta);
  __host__ __device__ float surfaceArea() const;
  __host__ __device__ float volumn() const;
  __host__ __device__ int maximumExtent() const;
  __host__ __device__ const Point& operator[](int i) const;

  Point p_min, p_max;
};

__host__ __device__ BBox unionBox(const BBox &b, const Point &p);
__host__ __device__ BBox unionBox(const BBox &b1, const BBox &b2);
#endif