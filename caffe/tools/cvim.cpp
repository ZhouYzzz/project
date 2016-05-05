#include <opencv2/opencv.hpp>

int main(int argc, char const *argv[])
{
    MAT A, C;
    A = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    return 0;
}