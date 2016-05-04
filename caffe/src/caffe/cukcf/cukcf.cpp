#include "caffe/cukcf/cukcf.hpp"

#include <gflags/gflags.h>

#include <vector>

#include "caffe/caffe.hpp"
#include "cufft.h"

#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

#define tic cudaEventRecord(start, 0);
#define toc cudaEventRecord(stop, 0); \
	cudaEventSynchronize(stop); \
	cudaEventElapsedTime(&time,start,stop);

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

		N=1;C=3;H=20;W=10;n=N*C*H*W; // init input shape
		size[0]=W;size[1]=H;
		
		CUFFT_CHECK(cufftPlanMany(&plan, 2, size,
			NULL, 1, H*W,
			NULL, 1, H*W,
			CUFFT_C2C, N*C)); // N*C batches of 2D metric of size H*W
		CUFFT_CHECK(cufftPlan2d(&planS, H, W, CUFFT_C2C));

		alphaf = initComplex(H*W);
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
		s_ = initComplex(H*W);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		time = 0.0;

		std::cout << " finished." << std::endl;
	}
	void test(); // for development

	void init(float* img, int* roi);
	void update(float* img);
	// params

protected:
	void train(cuComplex* x, cuComplex* y, float sigma, float lambda, cuComplex* dst);
	void detect(cuComplex* alphaf, cuComplex* x, cuComplex* z, float sigma, cuComplex* dst);

	void linearCorrelation(cuComplex* xf1, cuComplex* xf2, cuComplex* k);

	// gpu memory space initialization
	void allocate_mem_space(int N, int C, int H, int W);

	int id;
	int N, C, H, W, n; // input shape
	int size[2]; // single batch size
	int roi[2];
	cufftHandle plan; // fft plan
	cufftHandle planS; // fft plan single batch
	cuComplex* alphaf; // target model

	cublasHandle_t handle; // handle

	cudaEvent_t start,stop; // timer
	float time;
	float* resp;

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
	cuComplex* s_;
};

int main(int argc, char** argv) {
	//std::cout << "run cukcf" << std::endl;
	caffe::GlobalInit(&argc, &argv);
	Caffe::SetDevice(0);
	Caffe::set_mode(Caffe::GPU);

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
	//std::cout << "[tracker] linear correlation." << std::endl;

	CUFFT_CHECK(cufftExecC2C(plan, src1, src1, CUFFT_FORWARD));	// fft on x1, inplace
	cudaDeviceSynchronize();
	
	CUFFT_CHECK(cufftExecC2C(plan, src2, src2, CUFFT_FORWARD));	// fft on x2, inplace
	cudaDeviceSynchronize();
	
	caffe::caffe_gpu_mul_cjC(n, src1, src2, c_);				// c = conj(x1) .* x2, ref.
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
	CUFFT_CHECK(cufftExecC2C(planS, y, y, CUFFT_FORWARD)); // fft2(y)
	CUFFT_CHECK(cufftExecC2C(planS, kf_, kf_, CUFFT_FORWARD)); // fft2(k) single batch [NOTICE] may change
	caffe::caffe_gpu_add_scalar_C(H*W, kf_, make_cuFloatComplex(lambda, 0.0), s_); // s_ = fft2(k) + lambda
	caffe::caffe_gpu_div_C(H*W, y, s_, dst); // dst = fft2(y) ./ ( fft2(k) + lambda )
}

void Tracker::detect(cuComplex* alpha, cuComplex* x, cuComplex* z, float sigma, cuComplex* dst) {
	// function responses = detect(alphaf, x, z, sigma) 
	//     k = kernel_correlation(z, x, sigma);
	//     responses = real(ifft2(alphaf .* fft2(k)));
	// end
	linearCorrelation(z, x, kf_);

	CUFFT_CHECK(cufftExecC2C(planS, kf_, kf_, CUFFT_FORWARD)); // fft2(k)
	caffe::caffe_gpu_mul_C(H*W, kf_, alpha, s_); // s_ = fft2(k) .* alpha
	CUFFT_CHECK(cufftExecC2C(planS, alpha, s_, CUFFT_INVERSE)); // ifft2(s_)
	
	//resp =  complexToReal(s_, H*W, false);
}

void Tracker::test() {
	//	xf_ = initComplex(n);
	//	yf_ = initComplex(H*W);
	//	zf_ = initComplex(n);
	//	kf_ = initComplex(H*W);
    //
	//	x1f_ = initComplex(n);
	//	x2f_ = initComplex(n);
	//	c_ = initComplex(n);
	//	s_ = initComplex(H*W);
	tic;
	train(xf_, yf_, 0, 0, alphaf);
	toc;
	std::cout << "[test train] " << time << " ms" << std::endl;
	
	tic;
	detect(alphaf, zf_, xf_, 0, s_);
	toc;
	std::cout << "[test detect] " << time << " ms" << std::endl;
	std::cout << "[test] successful." << std::endl;
	return;
}

void Tracker::init(float* img, int* roi) {
	// roi: X,Y,H,W;
	roi_ = roi;
	// allocate_mem_space();
	tmpl_ = get_feature(img);
	prob_ = createGaussianPeak(H, W);

	train(tmpl_, prob_, 1, 0.0005, s_);
	return;
}

void Tracker::update(float* img) {
	newfeat_ = get_feature(img);
	float peak;
	res = detect(tmpl_, newfeat_, &peak);

	// adjust(res);
	// x = get_feature(newlocation)
	// train(x, factor)
	return;
}

void Tracker::allocate_mem_space(int N, int C, int H, int W) {
	return;
}
