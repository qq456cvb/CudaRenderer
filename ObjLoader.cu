
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

// line buffer size determines at compile time how large the input
// buffer should be for the file input lines
#define LINE_BUFFER_SIZE 1024

// shapes/Wavefront.cpp*
#include "ObjLoader.cuh"
#include <iostream>
// GETNUM just gets the next number from a line of input in an OBJ file
#ifndef GETNUM
#define GETNUM(lineBuffer, numBuffer, lindex, nindex, tval)  \
	nindex=0;\
	while ((lineBuffer[lindex] == ' ') || lineBuffer[lindex] == '/') lindex++;\
	while ((lineBuffer[lindex] != ' ') && (lineBuffer[lindex] != '/') && \
		   (lineBuffer[lindex] != '\0') && (lineBuffer[lindex] != '\n') && (lindex != LINE_BUFFER_SIZE)) { \
		numBuffer[nindex] = lineBuffer[lindex]; \
		nindex++; \
		lindex++; \
	} \
	numBuffer[nindex] = '\0'; \
	tval = atoi(numBuffer);
#endif

// constructor / parser
void
Wavefront::ParseOBJfile(string filename, int **vi, int *nvi, Point ** P, int * npi, Normal ** N, int * nni, float ** uvs, int *nuvi) {
  FILE* fin;
  fin = fopen(filename.c_str(), "r");
  if (!fin) {
    cout << strerror(errno) << endl;
    return;
  }

  // temporary input buffers
  vector<Point> points;
  vector<int> verts;
  vector<int> normalIndex;
  vector<int> uvIndex;

  vector<Normal> file_normals;
  vector<float> file_uvvector;

  Point ptmp;
  Normal ntmp;
  float uv1, uv2;

  char lineBuffer[LINE_BUFFER_SIZE];
  char numBuffer[256];
  int lindex = 0;
  int nindex = 0;
  int ival, uvval, nval;
  ntris = 0;

  // parse the data in
  while (fgets(lineBuffer, LINE_BUFFER_SIZE, fin)) {
    switch (lineBuffer[0]) {
    case 'v':
      // case vertex information
      if (lineBuffer[1] == ' ') {
        // regular vertex point
        sscanf(&lineBuffer[2], "%f %f %f", &ptmp.x, &ptmp.y, &ptmp.z);
        points.push_back(ptmp);
      }
      else if (lineBuffer[1] == 't') {
        // texture coordinates
        sscanf(&lineBuffer[3], "%f %f", &uv1, &uv2);
        file_uvvector.push_back(uv1);
        file_uvvector.push_back(uv2);
      }
      else if (lineBuffer[1] == 'n') {
        // normal vector
        sscanf(&lineBuffer[2], "%f %f %f", &ntmp.x, &ntmp.y, &ntmp.z);
        file_normals.push_back(ntmp);
      }
      break;
    case 'f':
      // case face information
      lindex = 2;
      ntris++;
      for (int i = 0; i < 3; i++) {

        GETNUM(lineBuffer, numBuffer, lindex, nindex, ival)

          // obj files go from 1..n, this just allows me to access the memory
          // directly by droping the index value to 0...(n-1)
          ival--;
        verts.push_back(ival);

        if (lineBuffer[lindex] == '/') {
          lindex++;
          GETNUM(lineBuffer, numBuffer, lindex, nindex, uvval)
            uvIndex.push_back(uvval - 1);
        }

        if (lineBuffer[lindex] == '/') {
          lindex++;
          GETNUM(lineBuffer, numBuffer, lindex, nindex, nval)
            normalIndex.push_back(nval - 1);
        }
        lindex++;
      }
      break;
    case 'g':
      // not really caring about faces or materials now
      // so just making life easier, I'll ignoring it
      break;
    }
  }

  fclose(fin);

  // merge everything back into one index array instead of multiple arrays
  MergeIndicies(points, file_normals, file_uvvector, verts, normalIndex, uvIndex);

  *vi = this->vertexIndex;
  *nvi = this->nvi;

  *P = this->p;
  *npi = *nvi;

  *N = this->n;
  *nni = *nvi;

  *uvs = this->uvs;
  *nuvi = *nvi;

  points.clear();
  file_normals.clear();
  file_uvvector.clear();
  verts.clear();
  normalIndex.clear();
  uvIndex.clear();


  printf("Found %d vertices\n", *nvi);
  printf("Found %d points\n", *nvi);

  if (n) printf("Used normal\n");
  if (uvs) printf("Used UVs\n");

}

void Wavefront::MergeIndicies(vector<Point> &points, vector<Normal> &normals, vector<float> &uvVec, vector<int> &vIndex, vector<int> &normalIndex, vector<int> &uvIndex) {

  bool useNormals = !normals.empty();
  bool useUVs = !uvVec.empty();


  if (!useNormals && !useUVs) {
    printf("Copying points\n");
    // just copy the points into the array
    nverts = vIndex.size();
    p = new Point[points.size()];
    nvi = points.size();
    for (unsigned int i = 0; i < points.size(); i++)
      p[i] = points[i];
    vertexIndex = new int[nverts];
    for (int i = 0; i < nverts; i++)
      vertexIndex[i] = vIndex[i];
    return;
  }

  // assumes that vertexIndex = normalIndex = uvIndex	
  nvi = nverts = vIndex.size();				// FIX: Dec, 3
  vertexIndex = new int[nverts];

  p = new Point[nverts];
  if (useNormals) n = new Normal[nverts];
  if (useUVs) uvs = new float[nverts * 2];

  for (int i = 0; i < nverts; i++) {
    p[i] = points[vIndex[i]];
    /*if (useNormals) n[i] = normals[normalIndex[i]];
    if (useUVs) {
      uvs[i * 2] = uvVec[uvIndex[i] * 2];
      uvs[i * 2 + 1] = uvVec[uvIndex[i] * 2 + 1];
    }*/
    vertexIndex[i] = i;
  }
}