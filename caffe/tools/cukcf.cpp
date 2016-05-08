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
DEFINE_string(weights, "net/metric_split_cuhk03/snapshots/train_val_iter_3000.caffemodel",
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

	Mat im = imread("database/cuhk03/labeled/0001_01.jpg", CV_LOAD_IMAGE_COLOR);
	LOG(INFO) << im.rows << " * " << im.cols;
	LOG(INFO) << CV_32F << " " << CV_8U;
	//Mat_<float> fm;
	//LOG(INFO) << fm.type();
	//im.convertTo(fm, CV_32F);
	//LOG(INFO) << fm.type();
	caffe::TransformationParameter param;
	//LOG(INFO) << param.mean_value_size() << " " << param.has_mean_file() ;
	LOG(INFO) << caffe::TEST;
	//caffe::DataTransformer<float> trans;
	//
	//
	caffe::Timer t;
	caffe::DataTransformer<float> trans(param, caffe::TEST);
	caffe::Blob<float> blob;
	blob.Reshape(1,im.channels(),im.rows,im.cols);
	LOG(INFO) << blob.shape(1);
	t.Start();
	trans.Transform(im, &blob);
	t.Stop();
	LOG(INFO) << t.MilliSeconds();
	LOG(INFO) << blob.cpu_data()[0];
	//tracker.cnn.input_blobs()[0]->mutable_cpu_data() = im;
	KCFTracker tracker(true,true,true,true);
	tracker.init(Rect(0,0,10,10), im);
	//LOG(INFO) << tracker.update(im);
	return 0;
}
