#include "Normal.cuh"

__host__ __device__ Normal::Normal() : x(0), y(0), z(0) {

}

__host__ __device__ Normal::Normal(const Normal &n) : x(n.x), y(n.y), z(n.z) {

}

__host__ __device__ Normal::Normal(const Vector &v) :
  x(v.x), y(v.y), z(v.z) {

}

__host__ __device__ Normal::Normal(float xx, float yy, float zz) :
  x(xx), y(yy), z(zz) {

}