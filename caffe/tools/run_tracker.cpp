#include <gflags/gflags.h>
#include <glog/logging.h>
#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "opencv2/opencv.hpp"

#include "cuComplex.h"
#include "cufft.h"

#include <cstdio>

DEFINE_string(model, "SqueezeNet_v1.0/feat.prototxt",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "SqueezeNet_v1.0/squeezenet_v1.0.caffemodel",
    "the pretrained weights to for testing.");
DEFINE_string(vedio, "bolt",
    "vedio squence.");

using namespace cv;
using namespace std;

Size2i get_search_window(Size2i, Size2i);

int main(int argc, char** argv)
{
    FLAGS_alsologtostderr = 1;

    caffe::GlobalInit(&argc, &argv);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(1);

    // caffe::Net<float> net(FLAGS_model, caffe::TEST);
    // net.CopyTrainedLayersFrom(FLAGS_weights);

    // ===================
    // Environment setting
    // ===================
	
    string vedio_path = "database/vot2013/"+FLAGS_vedio;
	char im_name[20]; sprintf(im_name, "/%08d.jpg", 1);
    Size2i target_sz(25, 60);
    Mat im = imread(vedio_path+string(im_name), CV_LOAD_IMAGE_COLOR);
    Size2i im_sz(im.cols, im.rows);
	LOG(INFO) << "target_sz:" << target_sz << "im_sz:" << im_sz;
    Size2i window_sz = get_search_window(target_sz, im_sz);
    LOG(INFO) << window_sz;
}

// ==============
// HELP FUNCTIONS
// ==============

Size2i get_search_window(Size2i target_sz, Size2i im_sz) {
    Size2i window_sz;
    if (target_sz.height / target_sz.width >= 2) {
        // large height
        window_sz.height = target_sz.height * 2;
        window_sz.width = target_sz.width * 2.8;
    } else if (im_sz.area() / target_sz.area() <= 10) {
        // large objects
        window_sz.height = target_sz.height * 2;
        window_sz.width = target_sz.width * 2;
    } else {
        // default
        window_sz.height = target_sz.height * 2.8;
        window_sz.width = target_sz.width * 2.8;
    }
    return window_sz;
}

cv::Mat createGaussianPeak(int H, int W) {
    cv::Mat_<float> res(H, W);
    int syh = (H) / 2; int sxh = (W) / 2;
    float output_sigma = sqrt((float) W * H) / 2 * 0.125;// padding, output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = exp(mult * (float) (ih * ih + jh * jh));
        }
    return res;
}

cv::Mat createHanningMats(int H, int W) {   
    cv::Mat hann1t = cv::Mat(cv::Size(H,1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,W), CV_32F, cv::Scalar(0)); 

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t;
    cv::Mat hann1d = hann2d.reshape(1,1); // Procedure do deal with cv::Mat multichannel bug
        
    hann = cv::Mat(cv::Size(size_patch[0]*size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
    for (int i = 0; i < size_patch[2]; i++) {
        for (int j = 0; j<size_patch[0]*size_patch[1]; j++) {
            hann.at<float>(i,j) = hann1d.at<float>(0,j);
        }
    }
    return hann;
}
