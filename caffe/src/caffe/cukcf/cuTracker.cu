#include <math_functions.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include "caffe/common.hpp"
#include "caffe/cukcf/cuTracker.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
__global__ void set_C_kernel(const int n, const cuComplex alpha, cuComplex* dst) {
	CUDA_KERNEL_LOOP(index, n) {
		dst[index] = alpha;
	}
}

void caffe_gpu_set_C(const int N, const cuComplex alpha, cuComplex* dst) {
	real_C_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, alpha, dst);
}

__global__ void real_C_kernel(const int n, const cuComplex* a, float* dst) {
	CUDA_KERNEL_LOOP(index, n) {
		dst[index] = cuCrealf(a[index]);
	}
}

void caffe_gpu_real_C(const int N, const cuComplex* a, float* dst) {
	real_C_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, a, dst);
}


__global__ void add_C_kernel(const int n, const cuComplex* a,
		const cuComplex* b, cuComplex* dst) {
	CUDA_KERNEL_LOOP(index, n) {
		dst[index] = cuCaddf(a[index], b[index]);
	}
}

void caffe_add_mul_C(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst) {
	add_C_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, a, b, dst);
}
__global__ void sub_C_kernel(const int n, const cuComplex* a,
		const cuComplex* b, cuComplex* dst) {
	CUDA_KERNEL_LOOP(index, n) {
		dst[index] = cuCsubf(a[index], b[index]);
	}
}

void caffe_gpu_sub_C(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst) {
	sub_C_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, a, b, dst);
}
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
