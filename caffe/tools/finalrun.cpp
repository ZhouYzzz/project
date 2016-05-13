#include "vot.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>

#include "caffe/caffe.hpp"
#include "caffe/cukcf/fast.hpp"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstdio>

DEFINE_string(model, "SqueezeNet_v1.0/feat.prototxt",
		    "The model definition protocol buffer text file.");
DEFINE_string(weights, "SqueezeNet_v1.0/squeezenet_v1.0.caffemodel",
		    "the pretrained weights to for testing.");

int main(int argc, char** argv) {
	
	FLAGS_alsologtostderr = 1;

	caffe::GlobalInit(&argc, &argv);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	caffe::Caffe::SetDevice(0);

	Fast tracker(FLAGS_model, FLAGS_weights);


	Mat im = imread("database/vot2013/bolt/00000001.jpg", CV_LOAD_IMAGE_COLOR);
	LOG(INFO) << im.rows << " * " << im.cols;
	
	tracker.init(Rect(336,165,25,60), im);
	
	char name[200];
	for (int i=2;i<100;i++) {
		snprintf(name, sizeof(char)*100, "database/vot2013/bolt/%08d.jpg", i);
		// LOG(INFO) << name;
		im = imread(name, CV_LOAD_IMAGE_COLOR);
		tracker.update(im);
		LOG(INFO) << tracker.get();
	}
	//LOG(INFO) << region->x;
	
	// VOTRegion region = vot.region();
	// string path = vot.frame();
	// VOTRegion region = vot.region();
	// cv::Mat image = cv::imread(vot.frame());
	// tracker.init(rect, image);

	// while (!vot.end()) {
	// 	string imagepath = vot.frame();
	// 	if (imagepath.empty()) break;
	// 	cv::Mat image = cv::imread(imagepath);
	// 	tracker.update(image);
	// 	vot.report(tracker.get());
	// }

	return 0;
}
