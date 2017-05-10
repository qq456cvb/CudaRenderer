#include "Point.cuh"
#include <stdio.h>

__host__ __device__ Point::Point() : x(0), y(0), z(0) {

}

__host__ __device__ Point::Point(float4 f) : x(f.x), y(f.y), z(f.z) {

}

__host__ __device__ Point::Point(const Point &p) : x(p.x), y(p.y), z(p.z) {

}

__host__ __device__ Point::Point(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {

}

__host__ __device__ Point Point::operator+(const Vector &v) const {
  return Point(x + v.x, y + v.y, z + v.z);
}

__host__ __device__ Point& Point::operator+=(const Vector &v) {
  x += v.x;
  y += v.y;
  z += v.z;
  return *this;
}

__host__ __device__ Point Point::operator-(const Vector &v) const {
  return Point(x - v.x, y - v.y, z - v.z);
}

__host__ __device__ Point& Point::operator-=(const Vector &v) {
  x -= v.x;
  y -= v.y;
  z -= v.z;
  return *this;
}

__host__ __device__ Vector Point::operator-(const Point &p) const {
  return Vector(x - p.x, y - p.y, z - p.z);
}

__host__ __device__ float Point::operator[](int idx) const {
  if (idx == 0)
  {
    return x;
  }
  else if (idx == 1) {
    return y;
  }
  else if (idx == 2)
  {
    return z;
  }
  else {
    printf("idx error in Point!\n");
    return 0;
  }
}

__host__ __device__ float distance(const Point &p1, const Point &p2) {
  return (p2 - p1).norm();
}

__host__ __device__ float distanceSquared(const Point &p1, const Point &p2) {
  float dist = (p2 - p1).norm();
  return dist * dist;
}

__host__ __device__ Point operator*(const Point &p, float f) {
  return Point(p.x * f, p.y *f, p.z * f);
}

__host__ __device__ Point operator*(float f, const Point &p) {
  return Point(p.x * f, p.y *f, p.z * f);
}

__host__ __device__ Point operator+(const Point &p1, const Point &p2) {
  return Point(p1.x + p2.x, p1.y + p2.y, p1.z + p2.z);
}