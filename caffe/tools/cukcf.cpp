#include <gflags/gflags.h>

#include <vector>

#include "caffe/caffe.hpp"
#include "cufft.h"

using caffe::Net;
using caffe::Caffe;
using caffe::Blob;
using caffe::vector;
DEFINE_string(model, "", "help");

int main(int argc, char** argv) {
  std::cout << "run cukcf" << std::endl;
  caffe::GlobalInit(&argc, &argv);

  Caffe::SetDevice(2);
  Caffe::set_mode(Caffe::GPU);
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  std::cout << "Net init done" << std::endl;
  const vector<Blob<float>*>& result = caffe_net.Forward();
  std::cout << result[0]->data_at(0,0,0,0) << std::endl;
  cufftReal* x = result[0]->mutable_gpu_data();
  return 0;
}
