#include <gflags/gflags.h>
#include <glog/logging.h>
#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "opencv2/opencv.hpp"

#include "cuComplex.h"
#include "cufft.h"

#include <cstdio>

#include "caffe/cukcf/recttools.hpp"

#define CUFFT_CHECK(condition) do{cufftResult result=condition;CHECK_EQ(result,CUFFT_SUCCESS)<<" "<<result;}while(0)

DEFINE_string(model, "test.prototxt",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "the pretrained weights to for testing.");

using namespace std;
using namespace cv;

cv::Mat createHanningMats(int C, int H, int W);
cv::Mat createGaussianPeak(int H, int W);

int main(int argc, char** argv)
{
    FLAGS_alsologtostderr = 1;

    caffe::GlobalInit(&argc, &argv);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(2);

    caffe::Net<float> cnn(FLAGS_model, caffe::TEST);
    // cnn.CopyTrainedLayersFrom(FLAGS_weights);

    cnn.Forward();
    int C, H, W, N;
    C = cnn.output_blobs()[0]->channels();
    H = cnn.output_blobs()[0]->height();
    W = cnn.output_blobs()[0]->width();
    N = C * H * W;
    
	// TEST mat
    Mat test(H, W, CV_32F);
	// LOG(INFO) << test;
    // TEST BEGIN

    cufftHandle planm_;
    cufftHandle plans_;
    cublasHandle_t handle_;

    float* tf1_;
    cuComplex* tm1_;
    cuComplex* tm2_;
    cuComplex* tm3_;
    cuComplex* ts1_;
    cuComplex* ts2_;
    cuComplex* ts3_;

    cuComplex one_;
    cuComplex zero_;
    cuComplex lambda_;
    cuComplex* ones_;
    cuComplex* null_;

    cuComplex* probf;
    cuComplex* alphaf;
    cuComplex* xf;
    cuComplex* zf;
    cuComplex* feat;
    cuComplex* hann;
    float* resp;

    one_ = make_cuFloatComplex(1.0, 0.0);
    zero_ = make_cuFloatComplex(0.0, 0.0);
    lambda_ = make_cuFloatComplex(0.0001, 0.0);
    CUDA_CHECK(cudaMalloc((void**)&null_, sizeof(cuComplex)*H*W));
    CUDA_CHECK(cudaMalloc((void**)&ones_, sizeof(cuComplex)*C));
    caffe::caffe_gpu_set_C(C, one_, ones_);

    CUDA_CHECK(cudaMalloc((void**)&feat, sizeof(cuComplex)*N));
    CUDA_CHECK(cudaMalloc((void**)&hann, sizeof(cuComplex)*N));

    CUDA_CHECK(cudaMalloc((void**)&probf, sizeof(cuComplex)*H*W));
    CUDA_CHECK(cudaMalloc((void**)&xf, sizeof(cuComplex)*N));
    CUDA_CHECK(cudaMalloc((void**)&zf, sizeof(cuComplex)*N));
    CUDA_CHECK(cudaMalloc((void**)&alphaf, sizeof(cuComplex)*H*W));
    CUDA_CHECK(cudaMalloc((void**)&resp, sizeof(float)*H*W));

    CUDA_CHECK(cudaMalloc((void**)&tf1_, sizeof(float)*N));
                                                            
    CUDA_CHECK(cudaMalloc((void**)&tm1_, sizeof(cuComplex)*N));
    CUDA_CHECK(cudaMalloc((void**)&tm2_, sizeof(cuComplex)*N));
    CUDA_CHECK(cudaMalloc((void**)&tm3_, sizeof(cuComplex)*N));

    CUDA_CHECK(cudaMalloc((void**)&ts1_, sizeof(cuComplex)*H*W));
    CUDA_CHECK(cudaMalloc((void**)&ts2_, sizeof(cuComplex)*H*W));
    CUDA_CHECK(cudaMalloc((void**)&ts3_, sizeof(cuComplex)*H*W));

    handle_ = caffe::Caffe::cublas_handle();
    int shape[2] = {H, W};
    CUFFT_CHECK(cufftPlanMany(&planm_, 2, shape, NULL, 1, H*W, NULL, 1, H*W, CUFFT_C2C, C));
    CUFFT_CHECK(cufftPlan2d(&plans_, H, W, CUFFT_C2C));

	Mat hann_ = createHanningMats(C, H, W);   
	Mat prob_ = createGaussianPeak(H, W);
	CUDA_CHECK(cudaMemcpy(tf1_, hann_.data, sizeof(float)*N, cudaMemcpyHostToDevice));
	caffe::caffe_gpu_cpy_R2C(N, tf1_, hann);
	CUDA_CHECK(cudaMemcpy(tf1_, prob_.data, sizeof(float)*H*W, cudaMemcpyHostToDevice));
	caffe::caffe_gpu_cpy_R2C(H*W, tf1_, probf);
	CUFFT_CHECK(cufftExecC2C(plans_, probf, probf, CUFFT_FORWARD));

	LOG(INFO) << "prob" << prob_;
	// TODO
	caffe::caffe_gpu_real_C(N, probf, tf1_);
    CUDA_CHECK(cudaMemcpy(test.data, tf1_, sizeof(float)*N, cudaMemcpyDeviceToHost));
    LOG(INFO) << "probf" << test;



    caffe::caffe_gpu_cpy_R2C(N, cnn.output_blobs()[0]->gpu_data(), tm1_);
	caffe::caffe_gpu_mul_C(N, tm1_, hann, feat);
	// TODO
	//caffe::caffe_gpu_real_C(N, tm1_, tf1_);
    //CUDA_CHECK(cudaMemcpy(test.data, tf1_, sizeof(float)*N, cudaMemcpyDeviceToHost));
    //LOG(INFO) << test;


	//caffe::caffe_gpu_mul_C(N, tm1_, ones_, feat);
    // CUDA_CHECK(cudaMemcpy(tm1_, feat, sizeof(cuComplex)*N, cudaMemcpyDeviceToDevice));

	// TODO
	caffe::caffe_gpu_real_C(N, hann, tf1_);
    CUDA_CHECK(cudaMemcpy(test.data, tf1_, sizeof(float)*N, cudaMemcpyDeviceToHost));
    LOG(INFO) << test;
	//
	
	// TODO
	caffe::caffe_gpu_real_C(N, feat, tf1_);
    CUDA_CHECK(cudaMemcpy(test.data, tf1_, sizeof(float)*N, cudaMemcpyDeviceToHost));
    LOG(INFO) << test;

	CUFFT_CHECK(cufftExecC2C(planm_, feat, xf, CUFFT_FORWARD)); // xf = fft(feat)
	
	// TODO
	//caffe::caffe_gpu_real_C(N, feat, tf1_);
    //CUDA_CHECK(cudaMemcpy(test.data, tf1_, sizeof(float)*N, cudaMemcpyDeviceToHost));
    //LOG(INFO) << "feat after fft" << test;

	// CUFFT_CHECK(cufftExecC2C(planm_, feat, xf, CUFFT_FORWARD)); // xf = fft(feat)

	caffe::caffe_gpu_mul_cjC(N, xf, xf, tm1_);
	// TODO
	caffe::caffe_gpu_real_C(N, tm1_, tf1_);
    CUDA_CHECK(cudaMemcpy(test.data, tf1_, sizeof(float)*N, cudaMemcpyDeviceToHost));
    LOG(INFO) << "conj(xf)*xf" << test;
	


	CUBLAS_CHECK(cublasCgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_T,
				1, H*W, C, &one_, ones_, C, tm1_, H*W, &zero_, ts1_, 1)); // ts1_ = k
	// TODO
	caffe::caffe_gpu_real_C(N, ts1_, tf1_);
    CUDA_CHECK(cudaMemcpy(test.data, tf1_, sizeof(float)*N, cudaMemcpyDeviceToHost));
    LOG(INFO) << "sum" << test;

	float f = 1.0/(N*H*W);
	CUBLAS_CHECK(cublasCsscal(handle_, H*W, &f, ts1_, 1));
	// TODO
	caffe::caffe_gpu_real_C(N, ts1_, tf1_);
    CUDA_CHECK(cudaMemcpy(test.data, tf1_, sizeof(float)*N, cudaMemcpyDeviceToHost));
    LOG(INFO) << "sum and scale" << test;

	caffe::caffe_gpu_add_scalar_C(H*W, ts1_, lambda_, ts2_); // fft(k)+lambda
	caffe::caffe_gpu_div_C(H*W, probf, ts2_, alphaf); // alphaf = probf / fft(k)+lambda
	// TODO
	caffe::caffe_gpu_real_C(N, alphaf, tf1_);
    CUDA_CHECK(cudaMemcpy(test.data, tf1_, sizeof(float)*N, cudaMemcpyDeviceToHost));
    LOG(INFO) << "alphaf" << test;

	// ok, now we can get new feature
	cnn.Forward();
    caffe::caffe_gpu_cpy_R2C(N, cnn.output_blobs()[0]->gpu_data(), tm1_);
	caffe::caffe_gpu_mul_C(N, tm1_, hann, feat);
	CUFFT_CHECK(cufftExecC2C(planm_, feat, feat, CUFFT_FORWARD)); // xf = fft(feat), now get zf

	caffe::caffe_gpu_mul_cjC(N, xf, feat, tm1_); // conj(xf).*zf
	CUBLAS_CHECK(cublasCgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_T,
				1, H*W, C, &one_, ones_, C, tm1_, H*W, &zero_, ts1_, 1)); // ts1_ = kzf = sum()

	CUBLAS_CHECK(cublasCsscal(handle_, H*W, &f, ts1_, 1)); //scale

	caffe::caffe_gpu_mul_C(H*W, ts1_, alphaf, ts2_);
	CUFFT_CHECK(cufftExecC2C(plans_, ts2_, ts2_, CUFFT_INVERSE));
	float fac = 1.0/H*W;
	CUBLAS_CHECK(cublasCsscal(handle_, H*W, &fac, ts2_, 1)); // scale by (1-factor)
// real part of ts2_ should be response

	// TODO
	caffe::caffe_gpu_real_C(N, ts2_, tf1_);
    CUDA_CHECK(cudaMemcpy(test.data, tf1_, sizeof(float)*N, cudaMemcpyDeviceToHost));
    LOG(INFO) << "resp" << test;
	

	

	LOG(INFO) << "ALLLLLLLLL";
}


cv::Mat createHanningMats(int C, int H, int W) {   
	cv::Mat hann1t = cv::Mat(cv::Size(H,1), CV_32F, cv::Scalar(0));
	cv::Mat hann2t = cv::Mat(cv::Size(1,W), CV_32F, cv::Scalar(0)); 
	for (int i = 0; i < H; i++)
		hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (W - 1)));
	for (int i = 0; i < W; i++)
		hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (H - 1)));
	cv::Mat hann2d = hann2t * hann1t;
	cv::Mat hann1d = hann2d.reshape(1,1); // Procedure do deal with cv::Mat multichannel bug       
	cv::Mat hann = cv::Mat(cv::Size(H*W, C), CV_32F, cv::Scalar(0));
	for (int i = 0; i < C; i++)
		for (int j = 0; j<H*W; j++)
			hann.at<float>(i,j) = hann1d.at<float>(0,j);
	return hann;
}

cv::Mat createGaussianPeak(int H, int W) {
	cv::Mat_<float> res(H, W);
	int syh = (H) / 2; int sxh = (W) / 2;
	float output_sigma = sqrt((float) W * H) / 2*0.2;// padding, output_sigma_factor;
	float mult = -0.5 / (output_sigma * output_sigma);
	for (int i = 0; i < H; i++)
		for (int j = 0; j < W; j++) {
			int ih = i - syh;
			int jh = j - sxh;
			res(i, j) = exp(mult * (float) (ih * ih + jh * jh));
		}
	return res;
}

