#include "Fresnel.cuh"

__host__ __device__ inline float cosTheta(const Vector &w) {
  return w.z;
}

__host__ __device__ FresnelDielectric::FresnelDielectric(float ei, float et) : eta_i(ei), eta_t(et) {

}


__host__ __device__ Vector FresnelDielectric::evaluate(float cosi) const {
  bool entering = cosi > 0.f;
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

  // Snell's law
  float sint = ei / et * sqrtf(std::fmaxf(0, 1.f - cosi * cosi));
  if (sint > 1.f)
  {
    // total reflection
    return 1.f;
  }

  float cost = sqrtf(std::fmaxf(0, 1.f - sint * sint));
  cosi = fabsf(cosi);

  float r_parl = (et * cosi - ei * cost) / (et * cosi + ei * cost);
  float r_perp = (ei * cosi - et * cost) / (ei * cosi + et * cost);

  return Vector((r_perp * r_perp + r_parl * r_parl) / 2.f);
}