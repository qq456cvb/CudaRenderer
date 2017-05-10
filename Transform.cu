#include "Transform.cuh"

__host__ __device__ Transform::Transform(const Matrix4x4 &mat) :
  m(mat), m_inv(mat.inverse()) {

}

__host__ __device__ Transform::Transform(const Matrix4x4 &mat, const Matrix4x4 &mat_inv) :
  m(mat), m_inv(mat_inv) {

}

__host__ __device__ Transform Transform::identity() {
  return Transform(Matrix4x4(1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1));
}

__host__ __device__ Transform Transform::inverse() const {
  return Transform(m_inv, m);
}

__host__ __device__ Transform Transform::translate(const Vector &delta) {
  Matrix4x4 mat(1, 0, 0, delta.x,
    0, 1, 0, delta.y,
    0, 0, 1, delta.z,
    0, 0, 0, 1);
  Matrix4x4 mat_inv(1, 0, 0, -delta.x,
    0, 1, 0, -delta.y,
    0, 0, 1, -delta.z,
    0, 0, 0, 1);
  return Transform(mat, mat_inv);
}

__host__ __device__ Transform Transform::scale(float x, float y, float z) {
  Matrix4x4 mat(x, 0, 0, 0,
    0, y, 0, 0,
    0, 0, z, 0,
    0, 0, 0, 1);
  Matrix4x4 mat_inv(1.f / x, 0, 0, 0,
    0, 1.f / y, 0, 0,
    0, 0, 1.f / z, 0,
    0, 0, 0, 1);
  return Transform(mat, mat_inv);
}

__host__ __device__ Transform Transform::rotateX(float angle) {
  float sin_t = sinf(angle);
  float cos_t = cosf(angle);
  Matrix4x4 mat(1, 0, 0, 0,
    0, cos_t, -sin_t, 0,
    0, sin_t, cos_t, 0,
    0, 0, 0, 1);
  return Transform(mat, mat.transpose());
}

__host__ __device__ Transform Transform::rotateY(float angle) {
  float sin_t = sinf(angle);
  float cos_t = cosf(angle);
  Matrix4x4 mat(cos_t, 0, -sin_t, 0,
    0, 1, 0, 0,
    sin_t, 0, cos_t, 0,
    0, 0, 0, 1);
  return Transform(mat, mat.transpose());
}

__host__ __device__ Transform Transform::rotateZ(float angle) {
  float sin_t = sinf(angle);
  float cos_t = cosf(angle);
  Matrix4x4 mat(cos_t, -sin_t, 0, 0,
    sin_t, cos_t, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1);
  return Transform(mat, mat.transpose());
}

__host__ __device__ Transform Transform::rotate(float angle, const Vector &axis) {
  Vector a = axis.normalized();
  float s = sinf(angle);
  float c = cosf(angle);
  float m[4][4];

  m[0][0] = a.x * a.x + (1.f - a.x * a.x) * c;
  m[0][1] = a.x * a.y * (1.f - c) - a.z * s;
  m[0][2] = a.x * a.z * (1.f - c) + a.y * s;
  m[0][3] = 0;

  m[1][0] = a.x * a.y * (1.f - c) + a.z * s;
  m[1][1] = a.y * a.y + (1.f - a.y * a.y) * c;
  m[1][2] = a.y * a.z * (1.f - c) - a.x * s;
  m[1][3] = 0;

  m[2][0] = a.x * a.z * (1.f - c) - a.y * s;
  m[2][1] = a.y * a.z * (1.f - c) + a.x * s;
  m[2][2] = a.z * a.z + (1.f - a.z * a.z) * c;
  m[2][3] = 0;

  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = 0;
  m[3][3] = 1;

  Matrix4x4 mat(m);
  return Transform(mat, mat.transpose());
}

__host__ __device__ Transform Transform::lookAt(const Point &pos, const Point &target, const Vector &up) {
  float m[4][4];
  // Initialize fourth column of viewing matrix
  m[0][3] = pos.x;
  m[1][3] = pos.y;
  m[2][3] = pos.z;
  m[3][3] = 1;

  // Initialize first three columns of viewing matrix
  Vector dir = (target - pos).normalized();
  if (up.normalized().cross(dir).norm() == 0) {
    printf("\"up\" vector (%f, %f, %f) and viewing direction (%f, %f, %f) "
      "passed to LookAt are pointing in the same direction.  Using "
      "the identity transformation.", up.x, up.y, up.z, dir.x, dir.y,
      dir.z);
    return Transform();
  }
  Vector right = up.normalized().cross(dir).normalized();
  Vector new_up = dir.cross(right);

  m[0][0] = right.x;
  m[1][0] = right.y;
  m[2][0] = right.z;
  m[3][0] = 0.;

  m[0][1] = new_up.x;
  m[1][1] = new_up.y;
  m[2][1] = new_up.z;
  m[3][1] = 0.;

  m[0][2] = dir.x;
  m[1][2] = dir.y;
  m[2][2] = dir.z;
  m[3][2] = 0.;

  Matrix4x4 cam_to_world(m);
  return Transform(cam_to_world.inverse(), cam_to_world);
}

__host__ __device__ Vector Transform::operator()(const Vector &v) const {
  float x = v.x, y = v.y, z = v.z;
  return Vector(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
    m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
    m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
}

__host__ __device__ Point Transform::operator()(const Point &p) const {
  float x = p.x, y = p.y, z = p.z;
  float px = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
  float py = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
  float pz = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
  float pw = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
  if (pw == 1.f)
  {
    return Point(px, py, pz);
  }
  else {
    if (pw == 0)
    {
      printf("Warning: point has zero w\n");
      return Point();
    }
    return Point(px / pw, py / pw, pz / pw);
  }

}
__host__ __device__ Ray Transform::operator()(const Ray &ray) const {
  Ray r;
  r.o = (*this)(ray.o);
  r.d = (*this)(ray.d);
  return r;
}

__host__ __device__ Normal Transform::operator()(const Normal &n) const {
  float x = n.x, y = n.y, z = n.z;
  return Normal(m_inv.m[0][0] * x + m_inv.m[1][0] * y + m_inv.m[2][0] * z,
    m_inv.m[0][1] * x + m_inv.m[1][1] * y + m_inv.m[2][1] * z,
    m_inv.m[0][2] * x + m_inv.m[1][2] * y + m_inv.m[2][2] * z);
}

__host__ __device__ Transform Transform::operator*(const Transform &t) const {
  Transform res;
  res.m = m * t.m;
  res.m_inv = t.m_inv * m_inv;
  return res;
}
