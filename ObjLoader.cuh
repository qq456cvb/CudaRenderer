
/*
pbrt source code Copyright(c) 1998-2010 Matt Pharr and Greg Humphreys.

This file is part of pbrt.

pbrt is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.  Note that the text contents of
the book "Physically Based Rendering" are *not* licensed under the
GNU GPL.

pbrt is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef PBRT_SHAPES_Wavefront_H
#define PBRT_SHAPES_Wavefront_H

// shapes/Wavefront.h*
#include "Point.cuh"
#include "Normal.cuh"
#include <map>
#include <string>
#include <vector>
using namespace std;
using std::map;

// Wavefront Declarations
class Wavefront {
public:
  // Wavefront Public Methods
  Wavefront() {}
  void ParseOBJfile(string filename, int **vi, int *nvi, Point ** P, int * npi, Normal ** N, int * nni, float ** uvs, int *nuvi);
  void MergeIndicies(vector<Point> &points, vector<Normal> &normals, vector<float> &uvVec, vector<int> &vIndex, vector<int> &normalIndex, vector<int> &uvIndex);

  int nvi, npi, nni;
  // TriangleMesh Proteted Data
  int ntris, nverts;
  int *vertexIndex;
  Point *p;
  Normal *n;
  Vector *s;
  float *uvs;

};

#endif // PBRT_SHAPES_Wavefront_H
