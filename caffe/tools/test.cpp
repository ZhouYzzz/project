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

int main(int argc, char const *argv[])
{
    FLAGS_alsologtostderr = 1;

    caffe::GlobalInit(&argc, &argv);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(1);

    caffe::Net<float> cnn(FLAGS_model, caffe::TEST);
    // cnn.CopyTrainedLayersFrom(FLAGS_weights);

    Net.Forward();
    int C, H, W, N;
    C = cnn.output_blobs()[0]->channels();
    H = cnn.output_blobs()[0]->height();
    W = cnn.output_blobs()[0]->width();
    N = C * H * W;

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

    // TEST mat
    Mat test(H, W, CV_32F, Scalar(C));
    LOG(INFO) << test.channels();
    // TEST BEGIN

    caffe::caffe_gpu_cpy_R2C(N, cnn.output_blobs()[0]->gpu_data(), tm1_);
    cudaMemcpy(test.data, tm1_, sizeof(float)*N, cudaMemcpyDeviceToHost);
    LOG(INFO) << test;
}