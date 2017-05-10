#include "ConstantTexture.cuh"
__host__ __device__ ConstantTexture::ConstantTexture() :
  color(Vector()) {

}

__host__ __device__ ConstantTexture::ConstantTexture(const Vector &c) :
  color(c) {

}


__host__ __device__ ConstantTexture::~ConstantTexture() {

}

__host__ __device__ Vector ConstantTexture::evaluate(const DifferentialGeometry &dg) const {
  return color;
}