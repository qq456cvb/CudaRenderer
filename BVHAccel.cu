#include "BVHAccel.cuh"
#include "CuTexture.cuh"
#include <assert.h>

//static texture<float4, 1, cudaReadModeElementType> node_texture;

__host__ __device__ inline void BVHAccel::swap(BVHPrimitiveInfo **b1, BVHPrimitiveInfo **b2) {
  BVHPrimitiveInfo *temp = *b1;
  *b1 = *b2;
  *b2 = temp;
}

__host__ __device__ BVHAccel::BVHPrimitiveInfo** BVHAccel::partition(BVHPrimitiveInfo** start, BVHPrimitiveInfo** end, int dim, float pmid) {
  BVHPrimitiveInfo** i = start, **j = end - 1, **k = start;
  while (k <= j) {
    if ((*k)->centroid[dim] < pmid)
    {
      swap(i++, k++);
    }
    else if ((*k)->centroid[dim] == pmid) {
      k++;
    }
    else {
      swap(k, j--);
    }
  }
  return k;
}

__host__ __device__ static inline bool intersectP(const BBox &bounds, const Ray &ray,
  const Vector &inv_dir, const int dir_is_neg[3]) {
  // Check for ray intersection against $x$ and $y$ slabs
  float tmin = (bounds[dir_is_neg[0]].x - ray.o.x) * inv_dir.x;
  float tmax = (bounds[1 - dir_is_neg[0]].x - ray.o.x) * inv_dir.x;
  float tymin = (bounds[dir_is_neg[1]].y - ray.o.y) * inv_dir.y;
  float tymax = (bounds[1 - dir_is_neg[1]].y - ray.o.y) * inv_dir.y;
  if ((tmin > tymax) || (tymin > tmax))
    return false;
  if (tymin > tmin) tmin = tymin;
  if (tymax < tmax) tmax = tymax;

  // Check for ray intersection against $z$ slab
  float tzmin = (bounds[dir_is_neg[2]].z - ray.o.z) * inv_dir.z;
  float tzmax = (bounds[1 - dir_is_neg[2]].z - ray.o.z) * inv_dir.z;
  if ((tmin > tzmax) || (tzmin > tmax))
    return false;
  if (tzmin > tmin)
    tmin = tzmin;
  if (tzmax < tmax)
    tmax = tzmax;
  return (tmin < ray.t_max) && (tmax > ray.t_min);
}

__host__ __device__ BVHAccel::BVHAccel(Primitive **p, int p_cnt)
  : max_prims_in_code(255)
{
  if (p_cnt > 0) {
    // memory leak may happen if not triangle mesh
    int refined_cnt = 0;
    printf("prims cnt: %d\n", refined_cnt);
    if (p_cnt == 1) {
      refined_cnt = (*p)->refine(primitives);
      printf("prims cnt: %d\n", refined_cnt);
    }
    else {
      int *cnt = new int[p_cnt];
      Primitive ***prims = new Primitive**[p_cnt];
      for (int i = 0; i < p_cnt; i++) {
        cnt[i] = p[i]->refine(prims[i]);
        refined_cnt += cnt[i];
      }
      primitives = new Primitive*[refined_cnt];
      Primitive **iter = primitives;
      for (int i = 0; i < p_cnt; i++) {
        memcpy(iter, prims[i], cnt[i] * sizeof(Primitive*));
        iter += cnt[i];
      }
      delete[] cnt;
      for (int i = 0; i < p_cnt; i++) {
        delete prims[i];
      }
      delete[] prims;
    }
    this->p_cnt = refined_cnt;
  }
  else {
    return;
  }

  // build BVH tree
  BVHPrimitiveInfo **build_data = new BVHPrimitiveInfo*[this->p_cnt];
  for (int i = 0; i < this->p_cnt; i++) {
    BBox bbox = primitives[i]->worldBound();
    build_data[i] = new BVHPrimitiveInfo(i, bbox);
  }
  int total_nodes = 0;
  int ordered_size = 0;
  Primitive **ordered_prims = new Primitive*[this->p_cnt];

  BVHBuildNode *root = nonRecursiveBuild(build_data, 0, this->p_cnt,
    total_nodes, ordered_prims, ordered_size);

  printf("total nodes: %d\n", total_nodes);
  node_num = total_nodes;

  Primitive **temp = primitives;
  primitives = ordered_prims;
  delete[] temp;
  // compact represention to match DFS access pattern
  nodes = new LinearBVHNode[total_nodes];
  int offset = 0;
  flattenBVHTree(root, offset);
  destroy(root);
  for (int i = 0; i < this->p_cnt; i++) {
    delete build_data[i];
  }
  delete[] build_data;
}

__host__ __device__ void BVHAccel::destroy(BVHBuildNode *node) {
  if (node->children[0]) {
    destroy(node->children[0]);
  }
  if (node->children[1]) {
    destroy(node->children[1]);
  }
  delete node;
}

__host__ __device__ BVHAccel::~BVHAccel()
{
}

__host__ __device__ BVHAccel::BVHBuildNode* BVHAccel::nonRecursiveBuild(BVHPrimitiveInfo **buildData, int start, int end,
  int &total_nodes, Primitive **ordered_prims, int &ordered_size) {
  //BVHBuildNode *nodes[64];
  const int stack_depth = 64;
  BVHBuildNode *parents[stack_depth];
  bool child_idxs[stack_depth];
  int starts[stack_depth];
  int ends[stack_depth];
    
  starts[0] = start;
  ends[0] = end;

  int todos = 1;
  BVHBuildNode *root = NULL;
  while (todos > 0) {
    total_nodes++;
    todos--;
    BVHBuildNode *parent = parents[todos];
    int child_idx = child_idxs[todos];
    int s = starts[todos];
    int e = ends[todos];

    BVHBuildNode *node = new BVHBuildNode();
    if (!root) {
      root = node;
    }
    if (root != node)
    {
      parent->children[child_idx] = node;
      node->parent = parent;
    }
    BBox bbox;
    for (int i = s; i < e; i++) {
      bbox = unionBox(bbox, buildData[i]->bounds);
    }
    int n_prims = e - s;
    if (n_prims == 1)
    {
      int first_prim_offset = ordered_size;
      for (int i = s; i < e; i++)
      {
        int prim_num = buildData[i]->prim_num;
        ordered_prims[ordered_size++] = primitives[prim_num];
      }
      node->initLeaf(first_prim_offset, n_prims, bbox);
      parent->bounds = unionBox(parent->bounds, node->bounds);
      if (child_idx == 0)
      {
        BVHBuildNode *iter = parent;
        while (iter->parent) {
          iter->parent->bounds = unionBox(iter->parent->bounds, iter->bounds);
          iter = iter->parent;
        }
      }
    }
    else {
      BBox centroid_bounds;
      for (int i = s; i < e; i++)
      {
        centroid_bounds = unionBox(centroid_bounds, buildData[i]->centroid);
        //printf("build data %d : %f, %f, %f\n", i, buildData[i].centroid.x, buildData[i].centroid.y, buildData[i].centroid.z);
      }
      int dim = centroid_bounds.maximumExtent();
      int mid = (s + e) / 2;

      // use mid point here
      float p_mid = 0.5f * (centroid_bounds.p_min[dim] + centroid_bounds.p_max[dim]);
      if (p_mid == centroid_bounds.p_min[dim] || p_mid == centroid_bounds.p_max[dim])
      {
        int first_prim_offset = ordered_size;
        for (int i = s; i < e; i++)
        {
          int prim_num = buildData[i]->prim_num;
          ordered_prims[ordered_size++] = primitives[prim_num];
        }
        node->initLeaf(first_prim_offset, n_prims, bbox);
        parent->bounds = unionBox(parent->bounds, node->bounds);
        if (child_idx == 0)
        {
          BVHBuildNode *iter = parent;
          while (iter->parent) {
            iter->parent->bounds = unionBox(iter->parent->bounds, iter->bounds);
            iter = iter->parent;
          }
        }
        continue;
      }
      BVHPrimitiveInfo **mid_ptr = partition(&buildData[s], &buildData[e], dim, p_mid);
      mid = mid_ptr - &buildData[0];

      //assert(mid != s && mid != e);
      parents[todos] = node;
      child_idxs[todos] = 0;
      starts[todos] = s;
      ends[todos] = mid;
      todos++;

      parents[todos] = node;
      child_idxs[todos] = 1;
      starts[todos] = mid;
      ends[todos] = e;
      todos++;

      node->split_axis = dim;
      node->n_prims = 0;
    }
  }
  return root;
}

__host__ __device__ BVHAccel::BVHBuildNode* BVHAccel::recursiveBuild(BVHPrimitiveInfo **buildData, int start, int end,
  int &total_nodes, Primitive **ordered_prims, int &ordered_size) {
  total_nodes++;
  BVHBuildNode *node = new BVHBuildNode();
  BBox bbox;
  for (int i = start; i < end; i++) {
    bbox = unionBox(bbox, buildData[i]->bounds);
  }
  int n_prims = end - start;
  if (n_prims == 1)
  {
    int first_prim_offset = ordered_size;
    for (int i = start; i < end; i++)
    {
      int prim_num = buildData[i]->prim_num;
      ordered_prims[ordered_size++] = primitives[prim_num];
    }
    node->initLeaf(first_prim_offset, n_prims, bbox);
  }
  else {
    BBox centroid_bounds;
    for (int i = start; i < end; i++)
    {
      centroid_bounds = unionBox(centroid_bounds, buildData[i]->centroid);
      //printf("build data %d : %f, %f, %f\n", i, buildData[i].centroid.x, buildData[i].centroid.y, buildData[i].centroid.z);
    }
    int dim = centroid_bounds.maximumExtent();
    int mid = (start + end) / 2;

    // use mid point here
    float p_mid = 0.5f * (centroid_bounds.p_min[dim] + centroid_bounds.p_max[dim]);
    if (p_mid == centroid_bounds.p_min[dim] || p_mid == centroid_bounds.p_max[dim]) {
      int first_prim_offset = ordered_size;
      for (int i = start; i < end; i++)
      {
        int prim_num = buildData[i]->prim_num;
        ordered_prims[ordered_size++] = primitives[prim_num];
      }
      node->initLeaf(first_prim_offset, n_prims, bbox);
      return node;
    }
    BVHPrimitiveInfo **mid_ptr = partition(&buildData[start], &buildData[end], dim, p_mid);
    mid = mid_ptr - &buildData[0];
    //assert(mid != start && mid != end);
    node->initInterior(dim,
      recursiveBuild(buildData, start, mid, total_nodes, ordered_prims, ordered_size),
      recursiveBuild(buildData, mid, end, total_nodes, ordered_prims, ordered_size));
  }
  return node;
}

__host__ __device__ int BVHAccel::flattenBVHTree(BVHBuildNode *node, int &offset) {
  LinearBVHNode *linear_node = &nodes[offset];
  linear_node->bounds = node->bounds;
  int curr_offset = offset++;
  if (node->n_prims > 0)
  {
    linear_node->primitive_offset = node->first_prim_offset;
    linear_node->n_prims = node->n_prims;
  }
  else { // interior
    linear_node->axis = node->split_axis;
    linear_node->n_prims = 0;
    flattenBVHTree(node->children[0], offset);
    linear_node->second_child_offset = flattenBVHTree(node->children[1], offset);
  }
  return curr_offset;
}

__host__ __device__ void BVHAccel::setSharedNodes(LinearBVHNode *nodes) {
  shared_nodes = nodes;
}

__host__ __device__ bool BVHAccel::intersect(const Ray &ray, Intersection *isect) const {
  /*if (!nodes)
  {
    return false;
  }*/
#ifdef __CUDA_ARCH__
  /*int idx = threadIdx.y * 32 + threadIdx.x;
  if (idx < node_num) {
    memcpy(&shared_nodes[idx], &nodes[idx], sizeof(LinearBVHNode));
  }*/
#endif
  bool hit = false;
  Point origin = ray.o;
  Vector inv_dir(1.f / ray.d.x, 1.f / ray.d.y, 1.f / ray.d.z);
  int dir_is_neg[3] = { inv_dir.x < 0, inv_dir.y < 0, inv_dir.z < 0 };

  int todo_offset = 0, node_num = 0;
  int todo[64];
  while (1)
  {
    const LinearBVHNode *node = &nodes[node_num];
    if (::intersectP(node->bounds, ray, inv_dir, dir_is_neg))
    {
      if (node->n_prims > 0)
      {
        for (int i = 0; i < node->n_prims; i++) {
          if (primitives[node->primitive_offset + i]->intersect(ray, isect))
          {
            hit = true;
          }
        }
        if (todo_offset == 0) break;
        node_num = todo[--todo_offset];
      }
      else {
        if (dir_is_neg[node->axis]) {
          todo[todo_offset++] = node_num + 1;
          node_num = node->second_child_offset;
        }
        else {
          todo[todo_offset++] = node->second_child_offset;
          node_num = node_num + 1;
        }
      }
    }
    else {
      if (todo_offset == 0) break;
      node_num = todo[--todo_offset];
    }
  }
  return hit;
}

__host__ __device__ void BVHAccel::getFlattenNodes(LinearBVHNode **nodes, int *len) {
  *nodes = this->nodes;
}

__host__ __device__ void BVHAccel::getBSDF(BSDF *bsdf, const DifferentialGeometry &dg,
  const Transform &obj_to_world) const {
  printf("BVHAccel getBSDF called!\n");
}

__host__ __device__ int BVHAccel::refine(Primitive **&prims) const {
  printf("BVHAccel refine called!\n");
  return 0;
}

__host__ __device__ BBox BVHAccel::worldBound() const {
  printf("BVHAccel worldBound called!\n");
  return BBox();
}