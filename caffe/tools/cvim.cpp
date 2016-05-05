#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main(int argc, char const *argv[])
{
	Mat A, C;
    A = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    return 0;
}
