#ifndef _COMMON_H_20240906_
#define _COMMON_H_20240906_

#define STRINGIFY(n) #n
#define TOSTRING(n) STRINGIFY(n)
#define __HERE__ __FILE__ " (" TOSTRING(__LINE__) "): "
#define PRINT_HERE mexPrintf(__HERE__);mexPrintf

#include "mex.h"
#include "tmwtypes.h"
#endif  // _COMMON_H_20240906_

#include "E:/github_upload/boslab-v2/Common/CUDA/types_BOSLAB.h"
#include "E:/github_upload/boslab-v2/Common/CUDA/GpuIds.hpp"

void freeGeoArray(unsigned int splits, Geometry* geoArray);
void checkFreeMemory(const GpuIds& gpuids, size_t *mem_GPU_global);
void printData(mwSize const numDims, const mwSize *size_flow, const Geometry& geo, const int diffselect, char* ptype, GpuIds &gpuids);
void printGeoData(const Geometry& geo);
void printProjectionsData(mwSize const numDims, const mwSize *size_flow);
