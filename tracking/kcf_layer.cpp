#include <vector>
#include <iostream>
#include "caffe/layers/kcf_layer.hpp"

namespace caffe {

template <typename Dtype>
void KCFLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  N_ = bottom[0]->num();
  C_ = bottom[0]->channels();
  H_ = bottom[0]->height();
  W_ = bottom[0]->width();
  int n[2] = {H_, W_};
  xf_.Reshape(N_, C_, H_, W_/2+1);
  if (cufftPlanMany(&plan_, 2, n,
			NULL, 1, 0,
			NULL, 1, 0,
			CUFFT_R2C, N_*C_) != CUFFT_SUCCESS) {
	fprintf(stderr, "CUFFT Error: Unable to create plan\n");
  }
  //cudaMalloc((void**)& xf_, sizeof(Complex)*H_*(W_/2+1));
  //cudaMalloc((void**)& xf_d, sizeof(cufftDoubleComplex)*H_*(W_/2+1));
  return;
}

template <typename Dtype>
void KCFLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //std::cout << "RESHAPE" << std::endl;
  vector<int> scalar(0);
  top[0]->Reshape(scalar);
  return;
}

template <typename Dtype>
void KCFLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //std::cout << "CF" << std::endl;
  top[0]->mutable_cpu_data()[0] = Dtype(1.0);
  return;
}

template <typename Dtype>
void KCFLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

#ifdef CPU_ONLY
STUB_GPU(KCFLayer);
#endif

INSTANTIATE_CLASS(KCFLayer);
REGISTER_LAYER_CLASS(KCF);

}  // namespace caffe
