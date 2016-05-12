#include "cuComplex.h"
#include "cufft.h"
#include "glog/logging.h"
#include <gflags/gflags.h>
#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>
#include "caffe/util/math_functions.hpp"
#define H 4
#define W 5
#define C 2
using namespace cv;

int main() {
	// FLAGS_alsologtostderr = 1;
	LOG(INFO) << "testing on cufft";
	
	cufftHandle plan;
	int nrank[2] = {H,W};
	cufftPlanMany(&plan, 2, nrank, NULL, 1, H*W, NULL, 1, H*W, CUFFT_C2C, C);
	
	float A[H][W] = {
		{0,0,0,0,0},
		{0,1,0,0,0},
		{0,0,0,0,0},
		{0,0,0,0,0}};
	
	Mat B = Mat::eye(H,W,CV_32F);

	LOG(INFO) << "\n" << B;

	float* Bg;

	cuComplex* Bc;

	cudaMalloc((void**)&Bg, sizeof(float)*H*W*C);
	cudaMalloc((void**)&Bc, sizeof(cuComplex)*H*W*C);

	cudaMemcpy(Bg, B.data, sizeof(float)*H*W, cudaMemcpyHostToDevice);
	cudaMemcpy(Bg+H*W, B.data, sizeof(float)*H*W, cudaMemcpyHostToDevice);
	
	caffe::caffe_gpu_cpy_R2C(H*W, Bg, Bc);


	// cufftExecC2C(plan, Bc, Bc, CUFFT_FORWARD);
	cufftExecC2C(plan, Bc, Bc, CUFFT_INVERSE);

	caffe::caffe_gpu_real_C(H*W*C, Bc, Bg);

	cudaMemcpy(B.data, Bg, sizeof(float)*H*W, cudaMemcpyDeviceToHost); LOG(INFO) << "\n" << B;
	
	return 0;
}
