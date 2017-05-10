#include "Specular.cuh"

__host__ __device__ inline float cosTheta(const Vector &w) {
  return w.z;
}

__host__ __device__ SpecularReflection::SpecularReflection(const Vector &r, Fresnel *f) :
  reflect(r), fresnel(f), BxDF(BxDFType(BSDF_REFLECTION | BSDF_SPECULAR)) {

}

__host__ __device__ SpecularReflection::~SpecularReflection() {
  if (fresnel)
  {
    delete fresnel;
    fresnel = NULL;
  }
}

__host__ __device__ Vector SpecularReflection::sample_f(const Vector &wo, Vector *wi, float *pdf) const {
  *wi = Vector(-wo.x, -wo.y, wo.z);
  *pdf = 1.f;
  return fresnel->evaluate(cosTheta(wo)) * reflect / fabsf(cosTheta(*wi));
}

__host__ __device__ Vector SpecularReflection::f(const Vector &wo, const Vector &wi) const {
  return Vector(0);
}
/**************************************************************/

__host__ __device__ SpecularTransmission::SpecularTransmission(const Vector &t, float ei, float et) :
  trans(t), fresnel(ei, et), BxDF(BxDFType(BSDF_TRANSMISSION | BSDF_SPECULAR)) {
  eta_i = ei;
  eta_t = et;
}

__host__ __device__ SpecularTransmission::~SpecularTransmission() {

}

__host__ __device__ Vector SpecularTransmission::sample_f(const Vector &wo, Vector *wi, float *pdf) const {
  bool entering = cosTheta(wo) > 0.f;
  float ei, float et;
  if (!entering)
  {
    ei = eta_t;
    et = eta_i;
  }
  else {
    ei = eta_i;
    et = eta_t;
  }

  float sini2 = std::fmaxf(0, 1.f - cosTheta(wo) * cosTheta(wo));
  float eta = ei / et;
  float sint2 = eta * eta * sini2;
  if (sint2 > 1.f)
  {
    return Vector(0);
  }

  float cost = sqrtf(std::fmaxf(0, 1.f - sint2));
  if (entering)
  {
    cost = -cost;
  }
  *wi = Vector(-eta * wo.x, -eta * wo.y, cost);
  *pdf = 1.f;
  Vector reflect_ratio = fresnel.evaluate(cosTheta(wo));
  return (et*et) / (ei*ei) * (Vector(1.f) - reflect_ratio) * trans
    / fabsf(cosTheta(*wi));
}

__host__ __device__ Vector SpecularTransmission::f(const Vector &wo, const Vector &wi) const {
  return Vector(0);
}