#include <gflags/gflags.h>
#include <glog/logging.h>

#include "caffe/caffe.hpp"
#include "caffe/cukcf/cuTracker.hpp"

#include "opencv2/opencv.hpp"

DEFINE_string(model, "net/metric_split_cuhk03/feat.prototxt",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "net/metric_split_cuhk03/snapshots/train_val_iter_3000.caffemodel",
    "the pretrained weights to for testing.");

using caffe::Caffe;
using namespace cv;

int main(int argc, char** argv)
{
    FLAGS_alsologtostderr = 1;

    caffe::GlobalInit(&argc, &argv);
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(2);

	// cuTracker tracker(FLAGS_model, FLAGS_weights);
	LOG(INFO) << "construct tracker done.";
	
	// LOG(INFO) << tracker.cnn.input_blobs()[0]->count();
	// LOG(INFO) << tracker.cnn.input_blobs()[0]->mutable_gpu_data();

	Mat im = imread("database/cuhk03/labeled/0001_01.jpg", CV_LOAD_IMAGE_COLOR);
	LOG(INFO) << im.rows << " * " << im.cols;
	return 0;
}
