#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/format.hpp"

#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "opencv2/opencv.hpp"

#include "cuComplex.h"
#include "cufft.h"

DEFINE_string(model, "SqueezeNet_v1.0/feat.prototxt",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "SqueezeNet_v1.0/squeezenet_v1.0.caffemodel",
    "the pretrained weights to for testing.");
DEFINE_string(vedio, "bolt",
    "vedio squence.");

using namespace cv;
using namespace std;

Size2i get_search_window(Size2i, Size2i, Size2i);

int main(int argc, char** argv)
{
    FLAGS_alsologtostderr = 1;

    caffe::GlobalInit(&argc, &argv);
    caffe::Caffe::set_mode(Caffe::GPU);
    caffe::Caffe::SetDevice(1);

    caffe::Net net(model, caffe::TEST);

    // ===================
    // Environment setting
    // ===================

    string vedio_path = boost::format{"database/vot2013/%s/"}%FLAGS_vedio;
    Size2i target_sz(25, 60);
    Mat im = imread(name, CV_LOAD_IMAGE_COLOR);
    Size2i im_sz(im.cols, im.rows);
    Size2i window_sz = get_search_window(target_sz, im_sz);
    LOG(INFO) << window_sz;
}

Size2i get_search_window(Size2i target_sz, Size2i im_sz) {
    Size_ window_sz();
    if (target_sz.height / target_sz.width > 2) {
        // large height
        window_sz.height = target_sz.height * 1.4;
        window_sz.width = target_sz.width * 2.8;
    } else if (im_sz.area() / target_sz.area() > 10) {
        // large objects
        window_sz.height = target_sz.height * 2;
        window_sz.width = target_sz.width * 2;
    } else {
        // default
        window_sz.height = target_sz.height * 2.8;
        window_sz.width = target_sz.width * 2.8;
    }
    return Size_<int>(window_sz);
}

