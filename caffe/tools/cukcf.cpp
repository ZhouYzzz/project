#include <gflags/gflags.h>

#include <vector>

#include "caffe/caffe.hpp"
#include "cufft.h"
#include "caffe/cukcf/cukcf.hpp"

#define tic cudaEventRecord(start, 0);
#define toc cudaEventRecord(stop, 0); \
	cudaEventSynchronize(stop); \
	cudaEventElapsedTime(&time,start,stop);


using caffe::Net;
using caffe::Caffe;
using caffe::Blob;
using caffe::vector;
DEFINE_string(model, "", "help");

#define CUFFT_CHECK(condition) \
	do { \
		cufftResult result = condition; \
		CHECK_EQ(result, CUFFT_SUCCESS) << " " \
		<< result; \
	} while (0)

cuComplex* realToComplex(const float* src, int n, bool fromdevice=true) {
	cuComplex* dst;
	// calloc mem space
	CUDA_CHECK(cudaMalloc((void**)&dst, sizeof(cuComplex)*n));
	CUDA_CHECK(cudaMemset(dst, 0, sizeof(cuComplex)*n));
	if (fromdevice) {
		CUDA_CHECK(cudaMemcpy2D(dst, sizeof(cuComplex), src, 
			sizeof(float), sizeof(float), n, cudaMemcpyDeviceToDevice));
	}else{
		CUDA_CHECK(cudaMemcpy2D(dst, sizeof(cuComplex), src, 
			sizeof(float), sizeof(float), n, cudaMemcpyHostToDevice));	
	}
	return dst;
}
float* complexToReal(const cuComplex* src, int n, bool tohost=true) {
	float* dst = (float*)std::calloc(n, sizeof(float));
	if (tohost) {
		CUDA_CHECK(cudaMemcpy2D(dst, sizeof(float), src,
			sizeof(cuComplex), sizeof(float), n, cudaMemcpyDeviceToHost));
	}else{
		CUDA_CHECK(cudaMemcpy2D(dst, sizeof(float), src,
			sizeof(cuComplex), sizeof(float), n, cudaMemcpyDeviceToDevice));
	}
	return dst;
}

cuComplex* initComplex(int n) {
	cuComplex* c;
	CUDA_CHECK(cudaMalloc((void**)&c, sizeof(cuComplex)*n));
	CUDA_CHECK(cudaMemset(c, 0, sizeof(cuComplex)*n));
	return c;
}

class Tracker {
public:
	Tracker(int id) {
		id = id;
		std::cout << "init Tracker ...";

		N=1;C=3;H=30;W=20;n=N*C*H*W; // init input shape
		size[0]=W;size[1]=H;
		
		CUFFT_CHECK(cufftPlanMany(&plan, 2, size,
			NULL, 1, H*W,
			NULL, 1, H*W,
			CUFFT_C2C, N*C)); // N*C batches of 2D metric of size H*W

		alphaf = initComplex(n);
		handle = Caffe::cublas_handle();

		one_ = make_cuFloatComplex(1.0, 0.0);
		zero_ = make_cuFloatComplex(0.0, 0.0);
		// init ones_
		float* ones;
		CUDA_CHECK(cudaMalloc((void**)&ones, sizeof(float)*n));
		CUDA_CHECK(cudaMemset(ones, 1, sizeof(float)*n));
		ones_ = realToComplex(ones, n);

		zeros_ = initComplex(H*W); // for sum purpose

		xf_ = initComplex(n);
		yf_ = initComplex(H*W);
		zf_ = initComplex(n);
		kf_ = initComplex(H*W);

		x1f_ = initComplex(n);
		x2f_ = initComplex(n);
		c_ = initComplex(n);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		time = 0.0;

		std::cout << " finished." << std::endl;
	}
	void test(); // for development

	void init(int* roi,float* img);
	void update(float* img);
	// params

protected:
	//void train(float* img, float factor);
	void train(cuComplex* x, cuComplex* y, float sigma, float lambda, cuComplex* dst);
	//void detect(float* img);
	void detect(cuComplex* alphaf, cuComplex* x, cuComplex* z, float sigma);

	void linearCorrelation(cuComplex* xf1, cuComplex* xf2, cuComplex* k);

	int id;
	int N, C, H, W, n; // input shape
	int size[2]; // single batch size
	int roi[2];
	cufftHandle plan; // fft plan
	cuComplex* alphaf; // target model

	cublasHandle_t handle; // handle

	cudaEvent_t start,stop; // timer
	float time;

private:
	cuComplex* ones_; // for sum purpose
	cuComplex* zeros_; // for sum purpose
	cuComplex  one_;
	cuComplex  zero_;

	cuComplex* xf_; // F of input feature
	cuComplex* yf_; 
	cuComplex* zf_;
	cuComplex* kf_;

	cuComplex* x1f_;
	cuComplex* x2f_;
	cuComplex* c_; // c = conj(x1f_) .* x2f_
};

int main(int argc, char** argv) {
	//std::cout << "run cukcf" << std::endl;
	caffe::GlobalInit(&argc, &argv);

	std::cout << "=== cukcf ===" << std::endl;
	Tracker tracker(1);
	tracker.test();

//	int N = 1;
//	int C = 3;
//	int H = 40;
//	int W = 30;
//	int n = N*C*H*W;
//	float* xh = (float*)std::calloc(n, sizeof(float));
//	xh[0] = 1; xh[1] = 2; xh[2] = 3;
//	cuComplex* xd = realToComplex(xh, n, false);
//	float* xh2 = complexToReal(xd, n);
//	std::cout << xh2[0] << " " << xh2[1] << " " << xh2[2] << std::endl;
//
//	cufftHandle plan;
//
//	int shape[2] = {W, H}; // Width is the inner most dim
//	CUFFT_CHECK(cufftPlanMany(&plan, 2, shape,
//			NULL, 1, H*W,
//			NULL, 1, H*W,
//			CUFFT_C2C, N*C)); // N*C batches of 2D metric of size H*W
//	
//	cufftComplex* xf = initComplex(n);
//
//	CUFFT_CHECK(cufftExecC2C(plan, xd, xf, CUFFT_FORWARD));
//
//	CUDA_CHECK(cudaDeviceSynchronize());
//
//	CUFFT_CHECK(cufftExecC2C(plan, xf, xd, CUFFT_INVERSE));
//	
//	float* xhf = complexToReal(xd, n);
//	std::cout << xhf[0] << " " << xhf[1] << " " << xhf[2] << std::endl;
	return 0;
}

void Tracker::linearCorrelation(cuComplex* src1, cuComplex* src2, cuComplex* dst) {
	std::cout << "[tracker] linear correlation." << std::endl;

	CUFFT_CHECK(cufftExecC2C(plan, src1, src1, CUFFT_FORWARD));	// fft on x1, inplace
	cudaDeviceSynchronize();
	
	CUFFT_CHECK(cufftExecC2C(plan, src2, src2, CUFFT_FORWARD));	// fft on x2, inplace
	cudaDeviceSynchronize();
	
	caffe::caffe_gpu_conj_mul(n, src1, src2, c_);				// c = conj(x1) .* x2, ref.
	cudaDeviceSynchronize();
	
	CUBLAS_CHECK(cublasCgemv(
				handle,		// sum c along Channels
				CUBLAS_OP_N,// transpose
				N*C,		// N
				H*W,		// M
				&one_,		// alpha = 1
				c_, N*C,		// N
				ones_, 1,	// M = N*C
				&zero_,		// beta = 0
				dst, 1		// output
				));
	cudaDeviceSynchronize();
	
	CUFFT_CHECK(cufftExecC2C(plan, dst, dst, CUFFT_INVERSE));		// ifft on k (H*W)
	cudaDeviceSynchronize();
}

void Tracker::train(cuComplex* x, cuComplex* y, float sigma, float lambda, cuComplex* dst) {
	// alpha = train(x, y, sigma, lambda) {
	//     k = kernel_correlation(x, x, sigma);
	//	   alphaf = fft2(y) ./ (fft2(k) + lambda);
	// }
	linearCorrelation(x, x, kf_);
	CUFFT_CHECK(cufftExecC2C(plan, y, y, CUFFT_FORWARD)); // fft2(y)
	CUFFT_CHECK(cufftExecC2C(plan, kf_, kf_, CUFFT_FORWARD)); // fft2(k)
	caffe::caffe_gpu_add_scalar_C(n, kf_, make_cuFloatComplex(lambda, 0.0), dst);
	
}

void Tracker::test() {
	tic;
	linearCorrelation(xf_, yf_, kf_);
	toc;
	std::cout << "[test] " << time << " ms" << std::endl;

	std::cout << "[test] successful." << std::endl;
	return;
}
