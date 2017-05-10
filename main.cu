#include "cpu.cuh"

//texture<int4, 1, cudaReadModeElementType> vi_texture;
//texture<float4, 1, cudaReadModeElementType> p_texture;
//cudaChannelFormatDesc channel_vi_desc, channel_p_desc;

__device__ float philips(float A, float2 k) {
  const float g = 9.8f;
  const float damp = 0.001f;
  const float2 w = { 64.f, 64.f };
  const float l2_w_2 = w.x * w.x + w.y * w.y;
  const float L = l2_w_2 / g;
  const float l = damp * L;
  const float L_2 = L * L;
  float l2_k_2 = k.x * k.x + k.y * k.y;
  float kw = k.x * w.x + k.y * w.y;

  return A * expf(-1.f / l2_k_2 / L_2 - l2_k_2*l*l)
    / l2_k_2 / l2_k_2
    *kw*kw / l2_k_2 / l2_w_2;
}

__global__ void initH0(float4 *h0, float4 *rand, int M, int N, int Lx, int Lz) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int m = x - M / 2;
  int n = y - N / 2;
  if (m == 0 && n == 0) return;

  float2 k;
  k.x = 2 * CUDART_PI * m / Lx;
  k.y = 2 * CUDART_PI * n / Lz;

  const float A = 0.005f;
  int idx = y * M + x;
  float cons = sqrtf(philips(A, k) / 2.f);
  h0[idx].x = rand[idx].x * cons;
  h0[idx].y = rand[idx].y * cons;
  h0[idx].z = rand[idx].z * cons;
  h0[idx].w = rand[idx].w * cons;
  if (x == 0)
  {
    //printf("x %d, y %d, val: %f %f %f %f\n", x, y, h0[idx].x, h0[idx].y, h0[idx].z, h0[idx].w);
  }
}

__global__ void getHk(float4 *h0, cuComplex *h_tilde, int M, int N, int Lx, int Lz, float t) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int m = x - M / 2;
  int n = y - N / 2;
  if (m == 0 && n == 0) {
    int idx = y * M + x;
    h_tilde[idx].x = h_tilde[idx].y = 0;
  }

  float2 k;
  k.x = 2 * CUDART_PI * m / Lx;
  k.y = 2 * CUDART_PI * n / Lz;

  const float g = 9.8f;
  int idx = y * M + x;
  float k_norm = sqrtf(k.x * k.x + k.y * k.y);
  float gkt = sqrtf(g * k_norm) * t;
  float cos_gkt = cosf(gkt);
  float sin_gkt = sinf(gkt);
  h_tilde[idx].x = (h0[idx].x + h0[idx].z) * cos_gkt - (h0[idx].y + h0[idx].w) * sin_gkt;
  h_tilde[idx].y = (h0[idx].y - h0[idx].w) * cos_gkt + (h0[idx].x - h0[idx].z) * sin_gkt;
}

__global__ void print(cuComplex * in, int M, int N) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int idx = y * M + x;
  float real = in[idx].x;
  float img = in[idx].y;
  if (x == M / 2 && y == N / 2)
  {
    printf("x %d, y %d, val: %f %f\n", x, y, real, img);
  }
}
__global__ void getAmp(cuComplex *in, Point *out, int M, int N) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int idx = y * M + x;
  // scale by data size due to cuFFT
  float real = in[idx].x / M / N;
  float img = in[idx].y / M / N;

  float trans = fminf(1.f, 0.2f * sqrtf(real * real + img * img));

  /*if (x == 0 && y == 0)
  {
    printf("Point: %f, %f, %f; trans:%f\n", out[idx].x, out[idx].y, out[idx].z, trans);
  }*/
  out[idx].y = trans;
}

inline float sampleNormal() {
  //return 0.5f;
  float x1, x2, w, y1, y2;

  do {
    x1 = 2.0 * ((float)rand() / (RAND_MAX)) - 1.0;
    x2 = 2.0 * ((float)rand() / (RAND_MAX)) - 1.0;
    w = x1 * x1 + x2 * x2;
  } while (w >= 1.0);

  w = sqrtf((-2.0 * logf(w)) / w);
  return x1 * w;
}

void cudaCheckError(cudaError_t err) {
  if (err != cudaSuccess)
  {
    printf("cuda error: %s\n", cudaGetErrorString(err));
    exit(0);
  }
}

__global__ void initScene(Scene *scene, int nvi, int npi, Point *p, int *vi) {
  Transform l2w = Transform::translate(Vector(0.3f, 1.f, 0));
  scene->lights = new PointLight(l2w, Vector(4.f));
  scene->n_lights = 1;
  
  Texture *tex = new ConstantTexture(Vector(200.f / 255.f, 150.f/255.f, 40.f/255.f));
  Material *material = new MatteMaterial(tex);
  /*Shape *sh = new Sphere(0.5f, 0.f, 2.f, 2.f * float(CUDART_PI_F));
  Transform trans = Transform::translate(Vector(0.2f, 0, 0));
  sh->shape_to_world = trans;
  sh->world_to_shape = trans.inverse();*/
  Transform o2w = Transform::identity();
  /*Transform o2w = Transform::scale(0.03f, 0.03f, 0.03f);
  o2w = o2w * Transform::rotateY(CUDART_PI / 2);
  o2w = o2w * Transform::rotateZ(3 * CUDART_PI / 4);*/
  Shape *sh = new TriangleMesh(o2w, o2w.inverse(), nvi / 4, npi, vi, p, NULL);
  Primitive *prim = new GeometricPrimitive(sh, material);
  Primitive **prims = nullptr;
  BVHAccel **accel = new BVHAccel*[1];
  accel[0] = new BVHAccel(&prim, 1);
  scene->n_prims = 1;
  scene->prims = (Primitive**)accel;
  //scene->n_prims = prim->refine(prims);
}

__global__ void initTracer(SamplerRenderer *renderer) {
  SurfaceIntegrator *integrator = new DirectLightingIntegrator(0);
  new(renderer) SamplerRenderer(integrator);
}

__global__ void rayTracer(Scene *scene, 
  SamplerRenderer *renderer,
  int width, int height, float*image) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    Vector dir(__int2float_rn(2 * x - width) / width, -1.f, __int2float_rn(2 * y - height) / width);
    dir.normalize();

    Transform t = Transform::rotateX(45.f / 180.f * CUDART_PI_F);
    Ray ray(Point(0, 1.f, 0), t(dir), 0);
    Vector v(0.f);

    //renderer->setSharedNodes(scene, shared_nodes);
    v = renderer->Li(scene, ray);

    image[3 * (y * width + x)] = min(v.x, 1.f);
    image[3 * (y * width + x) + 1] = min(v.y, 1.f);
    image[3 * (y * width + x) + 2] = min(v.z, 1.f);
}

#include <iostream>

void initSeaMesh(Point **p, int **vi, int *nvi, int *npi, int M, int N, float Lx, float Lz, float max_h) {
  float dx = Lx / (M - 1);
  float dz = Lz / (N - 1);

  *npi = M * N;
  *p = new Point[*npi];
  *nvi = (M - 1) * (N - 1) * 2 * 4;
  *vi = new int[*nvi];

  int cnt_1 = 0, cnt_2 = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int idx = j * M + i;
      (*p)[idx].x = dx * i - Lx / 2;
      if ((i + j) % 2)
      {
        (*p)[idx].y = max_h;
      }
      else {
        (*p)[idx].y = 0;
      }
      (*p)[idx].z = dz * j - Lz / 2;

      if (i < M - 1 && j < N - 1)
      {
        (*vi)[(j * (M - 1) + i) * 8] = idx;
        (*vi)[(j * (M - 1) + i) * 8 + 1] = idx + 1;
        (*vi)[(j * (M - 1) + i) * 8 + 2] = idx + M;

        (*vi)[(j * (M - 1) + i) * 8 + 4] = idx + M;
        (*vi)[(j * (M - 1) + i) * 8 + 5] = idx + 1;
        (*vi)[(j * (M - 1) + i) * 8 + 6] = idx + M + 1;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc > 1 && !strcmp(argv[1], "cpu"))
  {
    cpu_main();
    return;
  }
  // set stack size and heap size
  size_t size_heap, size_stack;
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 200000000);
  cudaDeviceSetLimit(cudaLimitStackSize, 16192);
  cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize);
  cudaDeviceGetLimit(&size_stack, cudaLimitStackSize);
  printf("Heap size found to be %d; Stack size found to be %d\n", (int)size_heap, (int)size_stack);

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  //cudaFuncSetCacheConfig(rayTracer, cudaFuncCachePreferL1);

  cudaError_t err;
  // load obj
  Wavefront wf;
  int *vi = NULL, nvi, npi, nni, nuvi;
  Point *p = NULL;
  Normal *n;
  /*float *uvs;
  wf.ParseOBJfile("C:\\Users\\Neil\\Downloads\\coin.obj", &vi, &nvi, &p, &npi, &n, &nni, &uvs, &nuvi);*/

  /* init sea*/
  int M = 1 << 6, N = 1 << 6;
  float Lx = 2.f, Lz = 2.f;
  initSeaMesh(&p, &vi, &nvi, &npi, M, N, Lx, Lz, 1.f);
  Lx *= 160.f;
  Lz *= 160.f;

  Point *p_device = NULL;
  int *vi_device = NULL;
  err = cudaMalloc((void **)&p_device, npi * sizeof(Point));
  err = cudaMalloc((void **)&vi_device, nvi * sizeof(int));
  cudaMemcpy(p_device, p, npi * sizeof(Point), cudaMemcpyHostToDevice);
  cudaMemcpy(vi_device, vi, nvi * sizeof(int), cudaMemcpyHostToDevice);

  /*channel_vi_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
  cudaBindTexture(NULL, &vi_texture, vi_device, &channel_vi_desc, nvi * sizeof(int));

  channel_p_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaBindTexture(NULL, &p_texture, p_device, &channel_p_desc, npi * sizeof(Point));*/

  srand(NULL);

  float4 *rand = (float4 *)malloc(M * N * sizeof(float4));
  for (int i = 0; i < M * N; i++) {
    rand[i].x = sampleNormal();
    rand[i].y = sampleNormal();
    rand[i].z = sampleNormal();
    rand[i].w = sampleNormal();
  }

  float4 *rand_d;
  cudaCheckError(cudaMalloc((void **)&rand_d, M * N * sizeof(float4)));
  cudaCheckError(cudaMemcpy(rand_d, rand, M * N * sizeof(float4), cudaMemcpyHostToDevice));

  float4 *h0;
  cudaCheckError(cudaMalloc((void **)&h0, M * N * sizeof(float4)));

  cuComplex *hk_comp;
  cudaCheckError(cudaMalloc((void **)&hk_comp, M * N * sizeof(cuComplex)));

  cufftHandle plan;
  cufftPlan2d(&plan, M, N, CUFFT_C2C);

  initH0 << <dim3(M / 32, N / 32), dim3(32, 32) >> > (h0, rand_d, M, N, Lx, Lz);
  float t = 0;
  /* init sea*/

    Scene *scene;
    cudaMalloc((void **)&scene, sizeof(Scene));
    initScene<<<1, 1>>>(scene, nvi, npi, p_device, vi_device);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    cout << cudaGetErrorString(err) << endl;

    SamplerRenderer *renderer;
    cudaMalloc((void **)&renderer, sizeof(SamplerRenderer));
    initTracer<<<1, 1 >>>(renderer);

// image generation
    unsigned int width = 640, height = 480;
    dim3 block_dim(32, 32);
    dim3 grid_dim((width - 1) / 32 + 1, (height - 1) / 32 + 1);

    float *image_device, *image_host;
    cudaMalloc((void **)&image_device, width * height * 3 * sizeof(float));
    image_host = (float *)malloc(width * height * 3 * sizeof(float));

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    cout << cudaGetErrorString(err) << endl;

    cv::Mat img(height, width, CV_32FC3, image_host, width * 3 * sizeof(float));
    std::chrono::time_point<std::chrono::steady_clock> begin, end;

    // code to benchmark
    
    while (1) {
      t += 0.1f;
      begin = std::chrono::high_resolution_clock::now();
      getHk << <dim3(M / 32, N / 32), dim3(32, 32) >> > (h0, hk_comp, M, N, Lx, Lz, t);
      cudaDeviceSynchronize();
      end = std::chrono::high_resolution_clock::now();
      std::cout << "get hk " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

      begin = std::chrono::high_resolution_clock::now();
      cufftExecC2C(plan, hk_comp, hk_comp, CUFFT_INVERSE);
      cudaDeviceSynchronize();
      end = std::chrono::high_resolution_clock::now();
      std::cout << "cuda fft " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

      begin = std::chrono::high_resolution_clock::now();
      getAmp << <dim3(M / 32, N / 32), dim3(32, 32) >> > (hk_comp, p_device, M, N);
      /*cudaDeviceSynchronize();
      err = cudaGetLastError();
      cout << cudaGetErrorString(err) << endl;*/
      cudaDeviceSynchronize();
      end = std::chrono::high_resolution_clock::now();
      std::cout << "get amp " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

      begin = std::chrono::high_resolution_clock::now();
      rayTracer << <grid_dim, block_dim >> >(scene, renderer, width, height, image_device);
      cudaDeviceSynchronize();
      end = std::chrono::high_resolution_clock::now();
      std::cout << "ray tracer + memcpy " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
      cudaMemcpy(image_host, image_device, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost);

      imshow("test", img);
      if ((char)cv::waitKey(10) == 27) break;
    }
    

    cudaFree(scene);
    cudaFree(renderer);
    cudaFree(image_device);
    cudaFree(p_device);
    cudaFree(vi_device);

    return 0;
}
