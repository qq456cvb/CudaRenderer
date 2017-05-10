#include "Matrix4x4.cuh"

__host__ __device__ Matrix4x4 Matrix4x4::identity() {
  return Matrix4x4(1.f, 0, 0, 0,
    0, 1.f, 0, 0,
    0, 0, 1.f, 0,
    0, 0, 0, 1.f);
}

__host__ __device__ Matrix4x4::Matrix4x4(const Matrix4x4 &mat) {
  m[0][0] = mat.m[0][0];
  m[0][1] = mat.m[0][1];
  m[0][2] = mat.m[0][2];
  m[0][3] = mat.m[0][3];

  m[1][0] = mat.m[1][0];
  m[1][1] = mat.m[1][1];
  m[1][2] = mat.m[1][2];
  m[1][3] = mat.m[1][3];

  m[2][0] = mat.m[2][0];
  m[2][1] = mat.m[2][1];
  m[2][2] = mat.m[2][2];
  m[2][3] = mat.m[2][3];

  m[3][0] = mat.m[3][0];
  m[3][1] = mat.m[3][1];
  m[3][2] = mat.m[3][2];
  m[3][3] = mat.m[3][3];
}

__host__ __device__ Matrix4x4::Matrix4x4(float a[4][4]) {
  m[0][0] = a[0][0];
  m[0][1] = a[0][1];
  m[0][2] = a[0][2];
  m[0][3] = a[0][3];

  m[1][0] = a[1][0];
  m[1][1] = a[1][1];
  m[1][2] = a[1][2];
  m[1][3] = a[1][3];

  m[2][0] = a[2][0];
  m[2][1] = a[2][1];
  m[2][2] = a[2][2];
  m[2][3] = a[2][3];

  m[3][0] = a[3][0];
  m[3][1] = a[3][1];
  m[3][2] = a[3][2];
  m[3][3] = a[3][3];
}

__host__ __device__ Matrix4x4::Matrix4x4(float t00, float t01, float t02, float t03,
  float t10, float t11, float t12, float t13,
  float t20, float t21, float t22, float t23,
  float t30, float t31, float t32, float t33) {
  m[0][0] = t00;
  m[0][1] = t01;
  m[0][2] = t02;
  m[0][3] = t03;

  m[1][0] = t10;
  m[1][1] = t11;
  m[1][2] = t12;
  m[1][3] = t13;

  m[2][0] = t20;
  m[2][1] = t21;
  m[2][2] = t22;
  m[2][3] = t23;

  m[3][0] = t30;
  m[3][1] = t31;
  m[3][2] = t32;
  m[3][3] = t33;
}


__host__ __device__ Matrix4x4 Matrix4x4::transpose() const {
  return Matrix4x4(m[0][0], m[1][0], m[2][0], m[3][0],
    m[0][1], m[1][1], m[2][1], m[3][1],
    m[0][2], m[1][2], m[2][2], m[3][2],
    m[0][3], m[1][3], m[2][3], m[3][3]);
}

__host__ __device__ void swapRow(float **a, float **b) {
  float *temp = *a;
  *a = *b;
  *b = temp;
}

__host__ __device__ void scaleRow(float *a, float scale, int size) {
  for (int i = 0; i < size; i++) {
    a[i] *= scale;
  }
}

__host__ __device__ void subRow(float *a, float *b, int size) {
  for (int i = 0; i < size; i++) {
    a[i] -= b[i];
  }
}

__host__ __device__ Matrix4x4 Matrix4x4::inverse() const {
  float rows[4][8];

  // Gauss-Jordan method
  for (int i = 0; i < 4; i++) {
    rows[i][0] = m[i][0]; rows[i][1] = m[i][1]; rows[i][2] = m[i][2]; rows[i][3] = m[i][3];
    rows[i][4] = i == 0 ? 1 : 0; rows[i][5] = i == 1 ? 1 : 0; rows[i][6] = i == 2 ? 1 : 0; rows[i][7] = i == 3 ? 1 : 0;
  }

  // upper triangle
  for (int i = 0; i < 4; i++) {
    bool non_zero = false;
    for (int j = i; j < 4; j++) {
      if (rows[j][i] != 0) {
        non_zero = true;
        swapRow((float **)&rows[j], (float **)&rows[i]);
        break;
      }
    }
    if (!non_zero) {
      printf("Matrix cannot be inverted!");
      return Matrix4x4();
    }
    for (int j = i + 1; j < 4; j++) {
      if (rows[j][i] != 0) {
        scaleRow(rows[j], rows[i][i] / rows[j][i], 8);
        subRow(rows[j], rows[i], 8);
      }
    }
  }

  // diagonal
  for (int i = 1; i < 4; i++) {
    for (int j = 0; j < i; j++) {
      if (rows[j][i] != 0)
      {
        scaleRow(rows[j], rows[i][i] / rows[j][i], 8);
        subRow(rows[j], rows[i], 8);
      }
    }
  }

  // identity
  for (int i = 0; i < 4; i++) {
    scaleRow(rows[i], 1.f / rows[i][i], 8);
  }

  return Matrix4x4(rows[0][4], rows[0][5], rows[0][6], rows[0][7],
    rows[1][4], rows[1][5], rows[1][6], rows[1][7],
    rows[2][4], rows[2][5], rows[2][6], rows[2][7],
    rows[3][4], rows[3][5], rows[3][6], rows[3][7]);
}

__host__ __device__ Matrix4x4 Matrix4x4::operator*(const Matrix4x4 &mat) const {
  Matrix4x4 r;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      r.m[i][j] = m[i][0] * mat.m[0][j] +
      m[i][1] * mat.m[1][j] +
      m[i][2] * mat.m[2][j] +
      m[i][3] * mat.m[3][j];
  return r;
}