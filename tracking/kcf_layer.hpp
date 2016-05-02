#ifndef CAFFE_KCF_LAYER_HPP_
#define CAFFE_KCF_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "cufft.h"
#include "vector_types.h"
#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

namespace caffe {

template <typename Dtype>
class KCFLayer: public Layer<Dtype> {
 public:
  explicit KCFLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "KCF"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  int N_, C_, H_, W_;

  cufftHandle plan_;
  cufftHandle iplan_;
  
  //cufftComplex* xf_;
  
  //cufftDoubleComplex* xf_d;
  Blob<cufftComplex> xf_;
//   typedef typename boost::conditional
// 	  <boost::is_same<Dtype,float>::value,
// 	  cufftComplex,cufftDoubleComplex>::type Complex;
//   Complex* xf_;
//   typedef typename boost::conditional
// 	  <boost::is_same<Dtype,float>::value,
//	  cufftReal,cufftDoubleReal>::type Real;
};

}

#endif
