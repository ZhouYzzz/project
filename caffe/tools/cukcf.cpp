#include <gflags/gflags.h>
#include <glog/logging.h>

#include "caffe/caffe.hpp"
#include "caffe/cukcf/cuTracker.hpp"
#include "caffe/cukcf/kcftracker.hpp"

#include "caffe/data_transformer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "opencv2/opencv.hpp"

DEFINE_string(model, "net/metric_split_cuhk03/feat.prototxt",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "net/metric_split_cuhk03/snapshots/train_val_iter_10000.caffemodel",
    "the pretrained weights to for testing.");

using caffe::Caffe;
using namespace cv;

int main(int argc, char** argv)
{
    FLAGS_alsologtostderr = 1;

    caffe::GlobalInit(&argc, &argv);
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(1);

	//cuTracker tracker(FLAGS_model, FLAGS_weights);
	LOG(INFO) << "construct tracker done.";
	
	// LOG(INFO) << tracker.cnn.input_blobs()[0]->count();
	// LOG(INFO) << tracker.cnn.input_blobs()[0]->mutable_gpu_data();

	caffe::TransformationParameter param;
	
	KCFTracker tracker(FLAGS_model, FLAGS_weights, param);
	Mat im = imread("database/MotorRolling/img/0001.jpg", CV_LOAD_IMAGE_COLOR);
	LOG(INFO) << im.rows << " * " << im.cols;
	
	tracker.init(Rect(117,68,122,125), im);
	
	im = imread("database/MotorRolling/img/0002.jpg", CV_LOAD_IMAGE_COLOR);
	LOG(INFO) << tracker.update(im);
	im = imread("database/MotorRolling/img/0003.jpg", CV_LOAD_IMAGE_COLOR);
	LOG(INFO) << tracker.update(im);
	im = imread("database/MotorRolling/img/0004.jpg", CV_LOAD_IMAGE_COLOR);
	LOG(INFO) << tracker.update(im);
	return 0;
}
