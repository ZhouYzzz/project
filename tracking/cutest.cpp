#include <gflags/gflags.h>

#include <vector>

#include "caffe/caffe.hpp"
#include "cufft.h"
#include "caffe/cukcf/cukcf.hpp"

using caffe::Net;
using caffe::Caffe;
using caffe::Blob;
using caffe::vector;
DEFINE_string(model, "", "help");

cuComplex* realToComplex(const float* src, int n, bool fromdevice=true) {
	cuComplex* dst;
	// calloc mem space
	cudaMalloc((void**)&dst, sizeof(cuComplex)*n);
	cudaMemset(dst, 0, sizeof(cuComplex)*n);
	if (fromdevice) {
		cudaMemcpy2D(dst, sizeof(cuComplex), src, 
			sizeof(float), sizeof(float), n, cudaMemcpyDeviceToDevice);
	}else{
		cudaMemcpy2D(dst, sizeof(cuComplex), src, 
			sizeof(float), sizeof(float), n, cudaMemcpyHostToDevice);	
	}
	return dst;
}
float* complexToReal(const cuComplex* src, int n, bool tohost=true) {
	float* dst;
	return dst;
}

int main(int argc, char** argv) {
  std::cout << "run cukcf" << std::endl;
  caffe::GlobalInit(&argc, &argv);

  Caffe::SetDevice(2);
  Caffe::set_mode(Caffe::GPU);
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  std::cout << "Net init done" << std::endl;
  const vector<Blob<float>*>& result = caffe_net.Forward();
  std::cout << result[0]->data_at(0,0,0,0) << std::endl;
  int N = result[0]->num();
  int C = result[0]->channels();
  int H = result[0]->height();
  int W = result[0]->width();
  std::cout <<N<<"*"<<C<<"*"<<H<<"*"<<W<<std::endl;


  size_t S_f = sizeof(float);
  float* xh = result[0]->mutable_cpu_data();
  std::cout <<xh[0]<<","<<xh[1]<<","<<xh[2]<<","<<xh[3]<<std::endl;

  cufftReal* x = result[0]->mutable_gpu_data();
  cufftComplex* xf;
  cufftComplex* yf;

  float* zeros = (float*)std::malloc(S_f*4);

  float* xh2 = (float*)std::malloc(S_f*8);
  cudaMemcpy(xh2, x, 4*S_f, cudaMemcpyDeviceToHost);
  int i=0;
  std::cout << "xh2: ";
  for (i=0;i<8;i++) {
	  std::cout << xh2[i] << ",";
  }
  std::cout << std::endl;

  cufftHandle plan;
  //cufftHandle iplan;

  //int in[2] = {W/2+1, H};
  cudaMalloc((void**)&xf, sizeof(cufftComplex)*4);

  cudaMemcpy(xh2, xf, 8*S_f, cudaMemcpyDeviceToHost);
  std::cout << "xh3 after create: ";
  for (i=0;i<8;i++) {
	  std::cout << xh2[i] << ",";
  }
  std::cout << std::endl;

  cudaEvent_t begin, end;
  cudaEventCreate(&begin); cudaEventCreate(&end);

  cudaEventRecord(begin,0);
  cudaMemset(xf, 0, 4*sizeof(cufftComplex));
  cudaMemcpy2D(xf, 2*S_f, x, S_f, S_f, C*H*W, cudaMemcpyDeviceToDevice);
  //cudaMemcpy2D(&xf[0].y, 2*S_f, zeros, S_f, S_f, C*H*W, cudaMemcpyHostToDevice);
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  float time;
  cudaEventElapsedTime(&time, begin, end);
  std::cout << "Memcpy:(ms) " << time << std::endl;

  cudaMemcpy(xh2, xf, 8*S_f, cudaMemcpyDeviceToHost);
  std::cout << "xh3 after cpy: ";
  for (i=0;i<8;i++) {
	  std::cout << xh2[i] << ",";
  }
  std::cout << std::endl;
  
  cudaEventRecord(begin,0);
  if (//cufftPlanMany(&plan, 2, n,
	//	  NULL, 1, H*W,
	//	  NULL, 1, H*W,
	//	  CUFFT_C2C, C*N)
	cufftPlan1d(&plan, W, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
	std::cout << "error1" << std::endl;
  }

  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, begin, end);
  std::cout << "Plan:(ms) " << time << std::endl;
  
  
  cudaEventRecord(begin,0);
  if (cufftExecC2C(plan, xf, xf, CUFFT_FORWARD) != CUFFT_SUCCESS) {
	std::cout << "error2" << std::endl;
  }
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, begin, end);
  std::cout << "fft:(ms) " << time << std::endl;
  cudaDeviceSynchronize();

  cudaMalloc((void**)&yf, sizeof(cufftComplex)*H*W);
  //if (cufftPlanMany(&plan, 2, in,
//		  NULL, 1, H*W,
//		  NULL, 1, H*W,
//		  CUFFT_C2C, C*N) != CUFFT_SUCCESS) {
//	std::cout << "error3" << std::endl;
  //}
  
  //for (i=0;i<1000;i++) {
  cudaEventRecord(begin,0);
  if (cufftExecC2C(plan, xf, yf, CUFFT_INVERSE) != CUFFT_SUCCESS) {
	std::cout << "error4" << std::endl;
  }
  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, begin, end);
  std::cout << "ifft:(ms) " << time << std::endl;
  cudaDeviceSynchronize();
  //}
  cudaMemcpy(xh2, yf, 8*S_f, cudaMemcpyDeviceToHost);
  std::cout << "xh3 fft ifft: ";
  for (i=0;i<8;i++) {
	  std::cout << xh2[i] << ",";
  }
  std::cout << std::endl;
  return 0;
}
