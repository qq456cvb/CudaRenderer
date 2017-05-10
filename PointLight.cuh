#ifndef CUDARENDERER_POINTLIGHT_H
#define CUDARENDERER_POINTLIGHT
#include "Light.cuh"

class PointLight : public Light
{
public:
  __host__ __device__ PointLight();
  __host__ __device__ PointLight(const Transform &l2w_, const Vector intensity_);
  __host__ __device__ ~PointLight();

  __host__ __device__ Vector sample_L(const Point &p, Vector *wi, float *pdf) const override;

  Point light_pos;
  Vector intensity;
private:

};



#endif