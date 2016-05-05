#include "caffe/caffe.hpp"
#include "caffe/cukcf/cuTracker.hpp"

DEFINE_string(model, "net/metric_split_cuhk03/feat.prototxt",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "net/metric_split_cuhk03/snapshots/train_val_iter_3000.caffemodel",
    "the pretrained weights to for testing.");

using caffe::Caffe;

int main(int argc, char** argv)
{
    // FLAGS_alsologtostderr = 1;

    caffe::GlobalInit(&argc, &argv);
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);

	cuTracker tracker(FLAGS_model, FLAGS_weights);
	return 0;
}
