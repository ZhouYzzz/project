#include "caffe/cukcf/cuTracker.hpp"

#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"
#include "caffe/util/math_functions.hpp"

using caffe::Caffe;
using caffe::Net;
using std::string;

void realToComplex(const float* src, int n, cuComplex* dst, bool fromdevice=true) {
	if (fromdevice) {
		CUDA_CHECK(cudaMemcpy2D(dst, sizeof(cuComplex), src, 
			sizeof(float), sizeof(float), n, cudaMemcpyDeviceToDevice));
	}else{
		CUDA_CHECK(cudaMemcpy2D(dst, sizeof(cuComplex), src, 
			sizeof(float), sizeof(float), n, cudaMemcpyHostToDevice));	
	}
}

#define CUFFT_CHECK(condition) \
	do { \
		cufftResult result = condition; \
		CHECK_EQ(result, CUFFT_SUCCESS) << " " \
		<< result; \
	} while (0)

cuTracker::cuTracker(string model, string weights) 
	: cnn(cnnInitCheck(model, weights), caffe::TEST) {
	init_constants();
	cnnLoad(weights);
}

void cuTracker::init(const cv::Rect &roi, const cv::Mat image) {
	pad = 2; // window size / roi size, constant

	img_H = image.rows; img_W = image.cols; 	// frame size
	roi_H = roi.height; roi_W = roi.width;		// roi size
	win_H = pad * roi_H; win_W = pad * roi_W; 	// window size

	// reshape, calculate feature size, get C, H, W
	cnn.input_blobs()[0]->Reshape(1,3,win_H,win_W);
	cnn.Reshape();
	Blob<float>* output = cnn.output_blobs()[0];
	C = output->channels();
	H = output->height();
	W = output->width();
	N = C * H * W;

	init_cuda_handle();
	allocate_memory_space();

	initHanning();
	initGaussian();

	getFeature(image); // to feat_ (cuComplex*)

	init_constants(1);
	train(feat_);
	init_constants(0.01);
}

cv::Rect cuTracker::update(cv::Mat image) {
	detect();
	update_location();
	getFeature();
	train();
	return cv::Rect();
}

string cuTracker::cnnInitCheck(string model, string weights) {
	CHECK_GT(model.size(), 0) << "Need a model definition.";
	CHECK_GT(weights.size(), 0) << "Need model weights.";
	return model;
}
void cuTracker::cnnLoad(string weights) {
	cnn.CopyTrainedLayersFrom(weights);
}

void cuTracker::allocate_memory_space() {
	using namespace caffe;
	// constants
	one_ = make_cuFloatComplex(1.0, 0.0);
	zero_ = make_cuFloatComplex(0.0, 0.0);
	CUDA_CHECK(cudaMalloc((void**)&null_, sizeof(cuComplex)*H*W));
	CUDA_CHECK(cudaMalloc((void**)&ones_, sizeof(cuComplex)*C));
	caffe_gpu_set_C(C, one_, ones_);

	// response
	CUDA_CHECK(cudaMalloc((void**)&resp_, sizeof(float)*H*W));

	// specified mem space
	CUDA_CHECK(cudaMalloc((void**)&tmpl_, sizeof(cuComplex)*N));
	CUDA_CHECK(cudaMalloc((void**)&feat_, sizeof(cuComplex)*N));

	CUDA_CHECK(cudaMalloc((void**)&alphaf_, sizeof(cuComplex)*H*W));
	CUDA_CHECK(cudaMalloc((void**)&k_, sizeof(cuComplex)*H*W));
	CUDA_CHECK(cudaMalloc((void**)&prob_, sizeof(cuComplex)*H*W));

	// temp mem space
	CUDA_CHECK(cudaMalloc((void**)&tm1_, sizeof(cuComplex)*N));
	CUDA_CHECK(cudaMalloc((void**)&tm2_, sizeof(cuComplex)*N));
	CUDA_CHECK(cudaMalloc((void**)&tm3_, sizeof(cuComplex)*N));

	CUDA_CHECK(cudaMalloc((void**)&ts1_, sizeof(cuComplex)*H*W));
	CUDA_CHECK(cudaMalloc((void**)&ts2_, sizeof(cuComplex)*H*W));
	CUDA_CHECK(cudaMalloc((void**)&ts3_, sizeof(cuComplex)*H*W));
}

void cuTracker::init_cuda_handle() {
	int shape[2] = {W, H};
	handle_ = Caffe::cublas_handle();
	CUFFT_CHECK(cufftPlanMany(&planm_, 2, shape,
			NULL, 1, H*W,
			NULL, 1, H*W,
			CUFFT_C2C, N)); // N*C batches of 2D metric of size H*W
	CUFFT_CHECK(cufftPlan2d(&plans_, H, W, CUFFT_C2C));
}

void cuTracker::init_constants(float interp) {
	lambda = make_cuFloatComplex(0.0001, 0);
	interp_factor = interp;
	interp_factor_C = make_cuFloatComplex(interp_factor, 0);
	onemin_factor = 1 - interp_factor;
}

void cuTracker::initHanning() {
	cv::Mat hann;
	cv::createHanningWindow(hann, cv::Size(H, W), CV_32F);
	CUDA_CHECK(cudaMemcpy(hann_, hann.data, H*W, cudaMemcpyHostToDevice));
}

void cuTracker::initGuassian() {
    cv::Mat_<float> res(H, W);

    int syh = (H) / 2;
    int sxh = (W) / 2;

    float output_sigma = std::sqrt((float) H * W) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
        }
    CUDA_CHECK(cudaMemcpy(prob_, res.data, H*W, cudaMemcpyHostToDevice));
    CUFFT_CHECK(cufftExecC2C(plans_, prob_, prob_, CUFFT_FORWARD)); // fft2(prob_) probf
}

cv::Rect cuTracker::getwindow(const cv::Rect roi) {
	pad = 2;
	float cx = roi.x + roi.width / 2;
    float cy = roi.y + roi.height / 2;
    cv::Rect window;
    window.width = pad*roi.width;
    window.height = pad*roi.height;
    window.x = cx - window.width / 2;
    window.y = cy - window.height / 2;
	return window;
}

void cuTracker::getFeature(const cv::Mat image) { // to feat_
	cv::Rect window = getwindow(roi_);
	cv::Mat z = RectTools::subwindow(image, window, cv::BORDER_REPLICATE);
	CUDA_CHECK(cudaMemcpy(cnn.input_blobs()[0]->mutable_gpu_data(), 
			z.data, sizeof(float) * win_H * win_W, cudaMemcpyHostToDevice));
	cnn.Forward();
	realToComplex(cnn.output_blobs()[0]->mutable_gpu_data(), N, feat_, true);

	// multi hanning window
	// !!!
}

void cuTracker::train(const cuComplex* x) { // interp_factor
	using namespace caffe;
	linearCorrelation(x, x, k_); // [note] fft(x) can only perform once
	// CUFFT_CHECK(cufftExecC2C(plans_, prob_, ts1_, CUFFT_FORWARD)); // ts1_ = fft2(y)
	CUFFT_CHECK(cufftExecC2C(plans_, k_, k_, CUFFT_FORWARD)); // k_ = fft2(k_)
	caffe_gpu_add_scalar_C(H*W, k_, lambda, ts2_); // ts2_ = fft2(k) + lambda
	caffe_gpu_div_C(H*W, prob_, ts2_, ts3_);	// alphaf = ts3_ = probf / ts_2

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
	CUBLAS_CHECK(cublasCgemv(handle_, CUBLAS_OP_T, H*W, C, &one_, tm3_, C, ones_, 1, &zero_, null_, 1));
	CUFFT_CHECK(cufftExecC2C(plans_, dst, dst, CUFFT_INVERSE));
}
