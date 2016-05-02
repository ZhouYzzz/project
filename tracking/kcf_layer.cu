#include <algorithm>
#include <vector>

#include "caffe/layers/kcf_layer.hpp"
#include "caffe/util/math_functions.hpp"
// #include "cufft.h"

namespace caffe {

template <>
void KCFLayer<float>::Forward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) {
  //cudaMalloc((void**)& xf_, sizeof(cufftComplex)*H_*(W_/2+1));
  cufftReal* x = bottom[0]->mutable_gpu_data();
  cufftComplex* xf = xf_.mutable_gpu_data();

  if (cufftExecR2C(plan_, x, xf) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
    return;
  }
  top[0]->mutable_cpu_data()[0] = float(1.0);
  return;
}

template <>
void KCFLayer<double>::Forward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
  //cudaMalloc((void**)& xf_d, sizeof(cufftDoubleComplex)*H_*(W_/2+1));
  //cufftDoubleReal* x = bottom[0]->mutable_gpu_data();
  //if (cufftExecR2C(plan_, x, xf_d) != CUFFT_SUCCESS){
  //  fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
  //  return;
  //}
  top[0]->mutable_cpu_data()[0] = double(1.0);
  return;
}

template <typename Dtype>
void KCFLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_LAYER_GPU_FUNCS(KCFLayer);

}  // namespace caffe
