#include <opencv2/opencv.hpp>
#include <iostream>
#include "glog/logging.h"
using namespace cv;

int main(int argc, char const *argv[])
{
	Mat A, C;
    A = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	LOG(INFO) << A;
    return 0;
}
