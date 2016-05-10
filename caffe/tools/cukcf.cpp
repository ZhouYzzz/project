#include <gflags/gflags.h>
#include <glog/logging.h>

#include "caffe/caffe.hpp"
#include "caffe/cukcf/cuTracker.hpp"
#include "caffe/cukcf/kcftracker.hpp"

#include "caffe/data_transformer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "opencv2/opencv.hpp"

#include <cstdio>
DEFINE_string(model, "SqueezeNet_v1.0/feat.prototxt",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "SqueezeNet_v1.0/squeezenet_v1.0.caffemodel",
    "the pretrained weights to for testing.");
//-model SqueezeNet_v1.0/feat.prototxt -weights SqueezeNet_v1.0/squeezenet_v1.0.caffemodel

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
	
	KCFTracker tracker(FLAGS_model, FLAGS_weights, param,true,true,false,true);
	Mat im = imread("database/vot2013/bolt/00000001.jpg", CV_LOAD_IMAGE_COLOR);
	LOG(INFO) << im.rows << " * " << im.cols;
	
	tracker.init(Rect(336,165,25,60), im);
	
//	caffe::Timer t;
//	t.Start();
//	im = imread("database/vot2013/bolt/00000001.jpg", CV_LOAD_IMAGE_COLOR);
//	LOG(INFO) << tracker.update(im);
//	t.Stop();
//	LOG(INFO) << "[TAKE]" << t.MilliSeconds() << "ms";
	
	int i;
	char name[200];
	for (i=2;i<100;i++) {
		snprintf(name, sizeof(char)*100, "database/vot2013/bolt/%08d.jpg", i);
		// LOG(INFO) << name;
		im = imread(name, CV_LOAD_IMAGE_COLOR);
		LOG(INFO) << i << "=============" << tracker.update(im);

	}
	return 0;
}
