#include <math_functions.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include "caffe/common.hpp"
#include "caffe/cukcf/cukcf.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

__global__ void conj_mul_kernel(const int n, const cuComplex* a,
		const cuComplex* b, cuComplex* dst) {
	CUDA_KERNEL_LOOP(index, n) {
		dst[index] = cuCmulf(cuConjf(a[index]), b[index]);
	}
}

void caffe_gpu_conj_mul(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst) {
	conj_mul_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
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


template <>
void caffe_gpu_gemv<cuComplex>(const CBLAS_TRANSPOSE TransA, const int M,
		const int N, const cuComplex alpha, const cuComplex* A, const cuComplex* x,
		const cuComplex beta, cuComplex* y) {
	cublasOperation_t cuTransA =
		(TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
	CUBLAS_CHECK(cublasCgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
		A, N, x, 1, &beta, y, 1));
}

}
