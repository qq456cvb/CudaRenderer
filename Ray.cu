#include "Ray.cuh"

__host__ __device__ Ray::Ray() :
  o(Point()), d(Vector()), t_min(0), t_max(INFINITY), depth(0) {

}

__host__ __device__ Ray::Ray(const Point &origin, const Vector &dir,
  float t_start, float t_end, int d) :
  o(origin), d(dir), t_min(t_start), t_max(t_end), depth(d) {
  if (fabsf(dir.norm() - 1) < 1e-4) {
    //        std::cout << "Warning: direction not normalized" << std::endl;
  }
}

__host__ __device__ Point Ray::operator()(float t) const {
  return o + t * d;
}