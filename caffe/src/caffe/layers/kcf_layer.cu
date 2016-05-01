#include <algorithm>
#include <vector>

#include "caffe/layers/kcf_layer.hpp"
#include "caffe/util/math_functions.hpp"
// #include "cufft.h"

namespace caffe {

template <typename Dtype>
void KCFLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->mutable_cpu_data()[0] = Dtype(1.0);
  return;
}

template <typename Dtype>
void KCFLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

INSTANTIATE_LAYER_GPU_FUNCS(KCFLayer);

}  // namespace caffe
