#define _USE_MATH_DEFINES

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <chrono>
#include "cufft.h"
#include "opencv2/opencv.hpp"
#include "Sphere.cuh"
#include "DifferentialGeometry.cuh"
#include "math_constants.h"
#include "Scene.cuh"
#include "PointLight.cuh"
#include "GeometricPrimitive.cuh"
#include "MatteMaterial.cuh"
#include "ConstantTexture.cuh"
#include "SamplerRenderer.cuh"
#include "DirectLightingIntegrator.cuh"
#include "TriangleMesh.cuh"
#include "ObjLoader.cuh"
#include "BVHAccel.cuh"

void initSeaMesh(Point **p, int **vi, int *nvi, int *npi, int M, int N, float Lx, float Lz, float max_h);
void cpu_main() {
  Wavefront wf;
  int *vi, nvi, npi, nni, nuvi;
  Point *p;
  Normal *n;
  float *uvs;
  //wf.ParseOBJfile("C:\\Users\\Neil\\Downloads\\yamato_simp.obj", &vi, &nvi, &p, &npi, &n, &nni, &uvs, &nuvi);

  int M = 1 << 5, N = 1 << 5;
  float Lx = 2.f, Lz = 2.f;
  initSeaMesh(&p, &vi, &nvi, &npi, M, N, Lx, Lz, 0.01f);

  Scene *scene = new Scene();
  Transform l2w = Transform::translate(Vector(0.3f, 1.f, 0));
  scene->lights = new PointLight(l2w, Vector(2.f));
  scene->n_lights = 1;

  Texture *tex = new ConstantTexture(Vector(1.f));
  Material *material = new MatteMaterial(tex);
  Transform o2w = Transform::scale(0.03f, 0.03f, 0.03f);
  o2w = o2w * Transform::rotateY(CUDART_PI / 2);
  o2w = o2w * Transform::rotateZ(CUDART_PI / 4);
  o2w = Transform::identity();
  /*Shape *sh = new Sphere(0.5f, 0.f, 2.f, 2.f * float(CUDART_PI_F));
  Transform trans = Transform::translate(Vector(0.2f, 0, 0));
  sh->shape_to_world = trans;
  sh->world_to_shape = trans.inverse();*/
  Shape *sh = new TriangleMesh(o2w, o2w.inverse(), nvi / 4, npi, vi, p, NULL);
  Primitive *prim = new GeometricPrimitive(sh, material);
  Primitive **prims = nullptr;
  BVHAccel *accel[1] = { new BVHAccel(&prim, 1) };

  
  scene->n_prims = 1;
  scene->prims = (Primitive**) accel;
  /*scene->n_prims = prim->refine(prims);
  scene->prims = prims;*/
  /*scene->n_prims = prim->refine(prims);
  scene->prims = prims;*/
  /*scene->prims = new GeometricPrimitive(sh, material);
  scene->n_prims = 1;*/

  int width = 640, height = 480;
  float *image;
  image = (float *)malloc(width * height * 3 * sizeof(float));

  SurfaceIntegrator *integrator = new DirectLightingIntegrator(0);
  SamplerRenderer *renderer = new SamplerRenderer(integrator);
  Transform t = Transform::rotateX(CUDART_PI_F / 4);

  std::chrono::time_point<std::chrono::steady_clock> begin, end;
  begin = std::chrono::high_resolution_clock::now();
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      Vector dir(float(2 * x - width) / width, -1.f, float(2 * y - height) / width);
      dir.normalize();

      Ray ray(Point(0, 1.f, 0), t(dir), 0);

      Vector v = renderer->Li(scene, ray);
      image[3 * (y * width + x)] = min(v.x, 1.f);
      image[3 * (y * width + x) + 1] = min(v.y, 1.f);
      image[3 * (y * width + x) + 2] = min(v.z, 1.f);
    }
  }
  
  end = std::chrono::high_resolution_clock::now();
  std::cout << "ray tracing " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;


  cv::Mat img(height, width, CV_32FC3, image, width * 3 * sizeof(float));
  cv::imshow("test", img);
  cv::waitKey(0);
}