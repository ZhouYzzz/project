#include "cufft.h"
#include "cuComplex.h"
#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"
namespace caffe {

void caffe_gpu_conj_mul(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst);

void caffe_gpu_add_scalar_C(const int N, const cuComplex* a, cuComplex alpha, 
		cuComplex* dst);
}
