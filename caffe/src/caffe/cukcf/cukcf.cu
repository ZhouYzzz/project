#include <math_functions.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include "caffe/common.hpp"
#include "caffe/cukcf/cukcf.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

__global__ void mul_C_kernel(const int n, const cuComplex* a,
		const cuComplex* b, cuComplex* dst) {
	CUDA_KERNEL_LOOP(index, n) {
		dst[index] = cuCmulf(a[index], b[index]);
	}
}

void caffe_gpu_mul_C(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst) {
	mul_C_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, a, b, dst);
}

__global__ void mul_cjC_kernel(const int n, const cuComplex* a,
		const cuComplex* b, cuComplex* dst) {
	CUDA_KERNEL_LOOP(index, n) {
		dst[index] = cuCmulf(cuConjf(a[index]), b[index]);
	}
}

void caffe_gpu_mul_cjC(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst) {
	mul_cjC_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, a, b, dst);
}

__global__ void add_scalar_C_kernel(const int n, const cuComplex* a,
		const cuComplex alpha, cuComplex* dst) {
	CUDA_KERNEL_LOOP(index, n) {
		dst[index] = cuCaddf(a[index], alpha);
	}
}

void caffe_gpu_add_scalar_C(const int N, const cuComplex* a, const cuComplex alpha, 
		cuComplex* dst) {
	add_scalar_C_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, a, alpha, dst);
}

__global__ void div_C_kernel(const int n, const cuComplex* a,
		const cuComplex* b, cuComplex* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = cuCdivf(a[index], b[index]);
	}
}

void caffe_gpu_div_C(const int N, const cuComplex* a,
		const cuComplex* b, cuComplex* dst) {
	// NOLINT_NEXT_LINE(whitespace/operators)
	div_C_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, a, b, dst);
}

}
