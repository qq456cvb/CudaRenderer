#ifndef CUDARENDERER_BVHACCEL_H
#define CUDARENDERER_BVHACCEL_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "Primitive.cuh"
#include "BBox.cuh"



struct LinearBVHNode {
  BBox bounds;
  union {
    int primitive_offset;
    int second_child_offset;
  };
  uint8_t n_prims;
  uint8_t axis;
  uint8_t padding[2];
};

class BVHAccel : public Primitive
{
  struct BVHPrimitiveInfo {
    __host__ __device__ BVHPrimitiveInfo() : prim_num(0) {}
    __host__ __device__ BVHPrimitiveInfo(int pn, const BBox &b)
      : prim_num(pn), bounds(b)
    {
      centroid = 0.5f * b.p_min + 0.5f * b.p_max;
    }
    __host__ __device__ ~BVHPrimitiveInfo() {}
    int prim_num;
    Point centroid;
    BBox bounds;
  };

  struct BVHBuildNode {
    __host__ __device__ BVHBuildNode() {
      children[0] = children[1] = NULL;
      parent = NULL;
    }

    __host__ __device__ ~BVHBuildNode() {}
    __host__ __device__ void initLeaf(int first, int n, const BBox &b) {
      first_prim_offset = first;
      n_prims = n;
      bounds = b;
    }

    __host__ __device__ void initInterior(int axis, BVHBuildNode *c1, BVHBuildNode *c2) {
      children[0] = c1;
      children[1] = c2;
      bounds = unionBox(c1->bounds, c2->bounds);
      split_axis = axis;
      n_prims = 0;
    }

    BVHBuildNode *parent;
    BVHBuildNode *children[2];
    BBox bounds;
    int split_axis, first_prim_offset, n_prims;
  };

  LinearBVHNode *nodes;
  LinearBVHNode *shared_nodes;
public:
  __host__ __device__ void getFlattenNodes(LinearBVHNode **nodes, int *len);
  __host__ __device__ BVHAccel(Primitive **p, int p_cnt);
  __host__ __device__ ~BVHAccel();

  __host__ __device__ BVHBuildNode* nonRecursiveBuild(BVHPrimitiveInfo **buildData, int start, int end,
    int &total_nodes, Primitive **ordered_prims, int &ordered_size);
  __host__ __device__ BVHBuildNode* recursiveBuild(BVHPrimitiveInfo **buildData, int start, int end,
    int &total_nodes, Primitive **ordered_prims, int &ordered_size);
  __host__ __device__ BVHPrimitiveInfo** partition(BVHPrimitiveInfo** start, BVHPrimitiveInfo** end, int dim, float pmid);
  __host__ __device__ inline void swap(BVHPrimitiveInfo **b1, BVHPrimitiveInfo **b2);

  __host__ __device__ int flattenBVHTree(BVHBuildNode *node, int &offset);

  __host__ __device__ bool intersect(const Ray &ray, Intersection *isect) const override;
  __host__ __device__ void setSharedNodes(LinearBVHNode *nodes) override;
  __host__ __device__ virtual void getBSDF(BSDF *bsdf, const DifferentialGeometry &dg,
    const Transform &obj_to_world) const override;
  __host__ __device__ virtual int refine(Primitive **&prims) const override;
  __host__ __device__ virtual BBox worldBound() const override;
private:
  __host__ __device__ void destroy(BVHBuildNode *node);
  Primitive **primitives;
  int p_cnt;
  int max_prims_in_code;
  int node_num;
};



#endif