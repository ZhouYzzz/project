#ifndef CAFFE_KCF_LAYER_HPP_
#define CAFFE_KCF_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
class KCFLayer: public Layer<Dtype> {
 public:
  explicit KCFLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Rshape(const vector<Blob<Dtype>*>& bottom,
      const vector<BLOB<Dtype>*>& top);

  virtual inline const char* type() const { return "KCF"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
};

}

#endif