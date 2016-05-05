#include "caffe/caffe.hpp"
#include "caffe/cukcf/cuTracker.hpp"

DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "the pretrained weights to for testing.");

int main(int argc, char** argv)
{
    FLAGS_alsologtostderr = 1;

    caffe::GlobalInit(&argc, &argv);
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);

	cuTracker tracker(FLAGS_model, FLAGS_weights);
	return 0;
}