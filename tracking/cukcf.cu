#include <algorithm>
#include <iostream>
#include "cukcf.hpp"

#include "cuComplex.h"
#include "cufft.h"

namespace CUKCF {


}

#define NX 8
#define BATCH 1

int main() {
  cuComplex c;
  c = make_cuComplex(1.0, 0.0);

  cufftHandle plan;
  cufftComplex *data;
  cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH);
  cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);
  cufftExecC2C(plan, data, data, CUFFT_FORWARD);
  std::cout << "Hello, cuda." << c.x << std::endl;
  return 0;
}
