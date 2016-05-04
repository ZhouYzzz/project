#include "cufft.h"
#include "cuComplex.h"
#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"
namespace caffe {

// a .* b
void caffe_gpu_mul_C(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst);

// conj(a) .* b
void caffe_gpu_mul_cjC(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst);

void caffe_gpu_add_scalar_C(const int N, const cuComplex* a, cuComplex alpha, 
		cuComplex* dst);

void caffe_gpu_div_C(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst);
}
