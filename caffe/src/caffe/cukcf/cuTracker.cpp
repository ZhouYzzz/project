#include "caffe/cukcf/cuTracker.hpp"

#include <gflags/gflags.h>

#include <vector>

#include "caffe/caffe.hpp"
#include "cufft.h"

#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

#define CUFFT_CHECK(condition) \
	do { \
		cufftResult result = condition; \
		CHECK_EQ(result, CUFFT_SUCCESS) << " " \
		<< result; \
	} while (0)

cuTracker::cuTracker() {

}

cuTracker::~cuTracker() {

}

void cuTracker::init(const cv::Rect &roi, cv::Mat image) {
	return;
}

cv::Rect cuTracker::update(cv::Mat image) {
	return cv::Rect();
}

void cuTracker::cnnLoad() {
	// load cnn from prototxt
	return;
}

void cuTracker::allocate_memory_space() {
	return;
}

void cuTracker::init_cuda_handle() {
	return;
}

void cuTracker::getFeature(const cv::Mat image, cuComplex* dst) {
	return;
}

void cuTracker::train(const cuComplex* x) { // interp_factor
	using namespace caffe;
	linearCorrelation(x, x, k_); // [note] fft(x) can only perform once
	CUFFT_CHECK(cufftExecC2C(plans_, prob_, ts1_, CUFFT_FORWARD)); // ts1_ = fft2(y)
	CUFFT_CHECK(cufftExecC2C(plans_, k_, k_, CUFFT_FORWARD)); // k_ = fft2(k_)
	caffe_gpu_add_scalar_C(H*W, k_, lambda, ts2_); // ts2_ = fft2(k) + lambda
	caffe_gpu_div_C(H*W, ts1_, ts2_, ts3_);	// alphaf = ts3_ = ts1_ / ts_2

	CUBLAS_CHECK(cublasCsscal(handle_, N, &onemin_factor, tmpl_, 1)); // scale by (1-factor)
	CUBLAS_CHECK(cublasCsscal(handle_, H*W,&onemin_factor, alphaf_, 1)); // scale by (1-factor)
	CUBLAS_CHECK(cublasCaxpy(handle_, N, &interp_factor_C,
				const_cast<cuComplex*>(x), 1, tmpl_, 1)); // tmpl_ += factor * x
	CUBLAS_CHECK(cublasCaxpy(handle_, H*W, &interp_factor_C,
				ts3_, 1, alphaf_, 1)); // alphaf_ += factor * alphaf
}

void cuTracker::detect(const cuComplex* z) { // resp_
	using namespace caffe;
	linearCorrelation(tmpl_, z, k_); // x = tmpl_, z: test image
	CUFFT_CHECK(cufftExecC2C(plans_, k_, k_, CUFFT_FORWARD)); // k_ = fft2(k_)
	caffe_gpu_mul_C(H*W, k_, alphaf_, ts1_); // ts1_ = alphaf_ .* fft2(k_)
	CUFFT_CHECK(cufftExecC2C(plans_, ts1_, ts1_, CUFFT_INVERSE)); // ts1_ = ifft2(ts1_)
	caffe_gpu_real_C(H*W, ts1_, resp_); // resp_ = real(ts1_)
}

void cuTracker::linearCorrelation(const cuComplex* a, const cuComplex* b, cuComplex* dst) {
	using namespace caffe;
	CUFFT_CHECK(cufftExecC2C(planm_, const_cast<cuComplex*>(a), tm1_, CUFFT_FORWARD));
	CUFFT_CHECK(cufftExecC2C(planm_, const_cast<cuComplex*>(b), tm2_, CUFFT_FORWARD));
	caffe_gpu_mul_cjC(N, tm1_, tm2_, tm3_);
	CUBLAS_CHECK(cublasCgemv(handle_, CUBLAS_OP_N, C, H*W, &one_, tm3_, C, ones_, 1, &zero_, dst, 1));
	CUFFT_CHECK(cufftExecC2C(plans_, dst, dst, CUFFT_INVERSE));
}
