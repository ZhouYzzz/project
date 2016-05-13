#include "caffe/caffe.hpp"
#include "opencv2/opencv.hpp"
#include "cufft.h"
#include "cuComplex.h"
#include "caffe/data_transformer.hpp"

using namespace std;
using namespace caffe;
using namespace cv;

class Fast {
public:
    Fast(string model, string weights);

    void init(const Rect &roi, Mat image);
    void update(Mat image);
    Rect get();

    Net<float> cnn;

    int C, H, W, N; // gpu size info
    float H_scal, W_scal;

private:
    Rect roi_;
	Size2i window_sz_;

    void init_handles();
    void init_constants();
    void init_mem_space();
    void init_hann_and_gaussian();
    Rect get_search_window_(const Rect &roi, Size2i window_sz);
    void extractFeature(const Rect &roi, Mat image);

    TransformationParameter trans_param;
    DataTransformer<float> trans;

protected:
    // handles
    cufftHandle planm_;
    cufftHandle plans_;
    cublasHandle_t handle_;

    // constants
    cuComplex one_;
    cuComplex zero_;
    cuComplex lambda_;
    cuComplex* ones_;
    cuComplex* null_;

    // mem space
    cuComplex* feat;
    cuComplex* hann;
    cuComplex* probf;
    cuComplex* xf;
    cuComplex* alphaf;
    cuComplex* model_xf;
    cuComplex* model_alphaf;
    // float* resp;
    Mat resp;

    // tmp mem space
    float* tf1_;
    cuComplex* tm1_;
    cuComplex* tm2_;
    cuComplex* ts1_;
    cuComplex* ts2_;
};
