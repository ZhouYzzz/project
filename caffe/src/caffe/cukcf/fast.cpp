#include "caffe/cukcf/fast.hpp"

#include "caffe/cukcf/recttools.hpp"

#define CUFFT_CHECK(condition) do { cufftResult result = condition; \
        CHECK_EQ(result, CUFFT_SUCCESS) << " " << result; } while (0)

using namespace caffe;
using namespace std;
using namespace cv;

cv::Mat createGaussianPeak_(int H, int W);
cv::Mat createHanningMats_(int C, int H, int W);

Fast::Fast(string model, string weights) 
: cnn(model, TEST), trans(trans_param, TEST)
{
    cnn.CopyTrainedLayersFrom(weights);
}

void Fast::init(const Rect &roi, Mat image)
{
    // get size
    Size2i window_sz(2.5*roi.height, 2.5*roi.width);
    window_sz_ = window_sz;
	LOG(INFO) << window_sz_;
    roi_ = roi;
    cnn.input_blobs()[0]->Reshape(1, 3, window_sz.height, window_sz.width);
    cnn.Reshape();
    C = cnn.output_blobs()[0]->channels();
    H = cnn.output_blobs()[0]->height();
    W = cnn.output_blobs()[0]->width();
    N = C * H * W;
    // scale factor due to cnn
    H_scal = float(window_sz.height) / H;
    W_scal = float(window_sz.width) / W;
	LOG(INFO) << H_scal << " " << W_scal;
	LOG(INFO) << C << " " << H << " " << W;

    // initialization
    init_handles();
    init_constants();
    init_mem_space();
    init_hann_and_gaussian();

    extractFeature(roi_, image);
    caffe::caffe_gpu_cpy_R2C(N, cnn.output_blobs()[0]->gpu_data(), tm1_);
    caffe::caffe_gpu_mul_C(N, tm1_, hann, feat);
    CUFFT_CHECK(cufftExecC2C(planm_, feat, model_xf, CUFFT_FORWARD));
    caffe::caffe_gpu_mul_cjC(N, model_xf, model_xf, tm1_);
    CUBLAS_CHECK(cublasCgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_T,
                1, H*W, C, &one_, ones_, C, tm1_, H*W, &zero_, ts1_, 1));
    float fm = 1.0 / N;
    CUBLAS_CHECK(cublasCsscal(handle_, H*W, &fm, ts1_, 1));
    caffe::caffe_gpu_add_scalar_C(H*W, ts1_, lambda_, ts2_);
    caffe::caffe_gpu_div_C(H*W, probf, ts2_, model_alphaf);
}

void Fast::update(Mat image)
{
    extractFeature(roi_, image);
    caffe::caffe_gpu_cpy_R2C(N, cnn.output_blobs()[0]->gpu_data(), tm1_);
    caffe::caffe_gpu_mul_C(N, tm1_, hann, feat);
    CUFFT_CHECK(cufftExecC2C(planm_, feat, feat, CUFFT_FORWARD));
    caffe::caffe_gpu_mul_cjC(N, model_xf, feat, tm1_);
    CUBLAS_CHECK(cublasCgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_T,
                1, H*W, C, &one_, ones_, C, tm1_, H*W, &zero_, ts1_, 1));
    float fm = 1.0 / N;
    CUBLAS_CHECK(cublasCsscal(handle_, H*W, &fm, ts1_, 1));    
    caffe::caffe_gpu_mul_C(H*W, ts1_, model_alphaf, ts2_);
    CUFFT_CHECK(cufftExecC2C(plans_, ts2_, ts2_, CUFFT_INVERSE));
    float fs = 1.0 / (H*W);
    CUBLAS_CHECK(cublasCsscal(handle_, H*W, &fs, ts2_, 1));
    caffe::caffe_gpu_real_C(N, ts2_, tf1_);
    CUDA_CHECK(cudaMemcpy(resp.data, tf1_, sizeof(float)*H*W, cudaMemcpyDeviceToHost));

    Point2i pi;
    double pv;
    cv::minMaxLoc(resp, NULL, &pv, NULL, &pi);
    float peak_value = (float) pv;
    // fine estimate
	LOG(INFO) << pi;
    roi_.x += W_scal * (pi.x - W / 2);
    roi_.y += H_scal * (pi.y - H / 2);

    extractFeature(roi_, image);
    caffe::caffe_gpu_cpy_R2C(N, cnn.output_blobs()[0]->gpu_data(), tm1_);
    caffe::caffe_gpu_mul_C(N, tm1_, hann, feat);
    CUFFT_CHECK(cufftExecC2C(planm_, feat, xf, CUFFT_FORWARD));
    caffe::caffe_gpu_mul_cjC(N, xf, xf, tm1_);
    CUBLAS_CHECK(cublasCgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_T,
                1, H*W, C, &one_, ones_, C, tm1_, H*W, &zero_, ts1_, 1));
    CUBLAS_CHECK(cublasCsscal(handle_, H*W, &fm, ts1_, 1));
    caffe::caffe_gpu_add_scalar_C(H*W, ts1_, lambda_, ts2_);
    caffe::caffe_gpu_div_C(H*W, probf, ts2_, alphaf);

    // model update
    float train_interp_factor = 0.01;
    float one_min_factor = 1 - train_interp_factor;
    cuComplex factor = make_cuFloatComplex(train_interp_factor, 0);

    CUBLAS_CHECK(cublasCsscal(handle_, N, &one_min_factor, model_xf, 1));
    CUBLAS_CHECK(cublasCsscal(handle_, H*W,&one_min_factor, model_alphaf, 1));
    CUBLAS_CHECK(cublasCaxpy(handle_, N, &factor, xf, 1, model_xf, 1));
    CUBLAS_CHECK(cublasCaxpy(handle_, H*W, &factor, alphaf, 1, model_alphaf, 1));
}

Rect Fast::get() { return roi_; }

void Fast::init_handles()
{
    handle_ = caffe::Caffe::cublas_handle();
    int shape[2] = {H, W};
    CUFFT_CHECK(cufftPlanMany(&planm_, 2, shape, NULL, 1, H*W, NULL, 1, H*W, CUFFT_C2C, C));
    CUFFT_CHECK(cufftPlan2d(&plans_, H, W, CUFFT_C2C));
    return;
}

void Fast::init_constants()
{
    one_ = make_cuFloatComplex(1.0, 0.0);
    zero_ = make_cuFloatComplex(0.0, 0.0);
    lambda_ = make_cuFloatComplex(0.0001, 0.0);
    CUDA_CHECK(cudaMalloc((void**)&null_, sizeof(cuComplex)*H*W));
    CUDA_CHECK(cudaMalloc((void**)&ones_, sizeof(cuComplex)*C));
    caffe::caffe_gpu_set_C(C, one_, ones_);
    return;
}

void Fast::init_mem_space()
{
    CUDA_CHECK(cudaMalloc((void**)&feat, sizeof(cuComplex)*N));
    CUDA_CHECK(cudaMalloc((void**)&hann, sizeof(cuComplex)*N));
    CUDA_CHECK(cudaMalloc((void**)&probf, sizeof(cuComplex)*H*W));
    CUDA_CHECK(cudaMalloc((void**)&xf, sizeof(cuComplex)*N));
    CUDA_CHECK(cudaMalloc((void**)&model_xf, sizeof(cuComplex)*N));
    CUDA_CHECK(cudaMalloc((void**)&alphaf, sizeof(cuComplex)*H*W));
    CUDA_CHECK(cudaMalloc((void**)&model_alphaf, sizeof(cuComplex)*H*W));
    CUDA_CHECK(cudaMalloc((void**)&resp, sizeof(float)*H*W));

    CUDA_CHECK(cudaMalloc((void**)&tf1_, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc((void**)&tm1_, sizeof(cuComplex)*N));
    CUDA_CHECK(cudaMalloc((void**)&tm2_, sizeof(cuComplex)*N));
    CUDA_CHECK(cudaMalloc((void**)&ts1_, sizeof(cuComplex)*H*W));
    CUDA_CHECK(cudaMalloc((void**)&ts2_, sizeof(cuComplex)*H*W));

	Mat resp_(H, W, CV_32F);
	resp = resp_;
}

void Fast::init_hann_and_gaussian()
{
    Mat hann_ = createHanningMats_(C, H, W);
    CUDA_CHECK(cudaMemcpy(tf1_, hann_.data, sizeof(float)*N, cudaMemcpyHostToDevice));
    caffe::caffe_gpu_cpy_R2C(N, tf1_, hann);

    Mat prob_ = createGaussianPeak_(H, W);
    CUDA_CHECK(cudaMemcpy(tf1_, prob_.data, sizeof(float)*H*W, cudaMemcpyHostToDevice));
    caffe::caffe_gpu_cpy_R2C(H*W, tf1_, probf);
    CUFFT_CHECK(cufftExecC2C(plans_, probf, probf, CUFFT_FORWARD));
}

Rect Fast::get_search_window_(const Rect &roi, Size2i window_sz)
{
    Rect window;
    int cx = roi.x - roi.width / 2;
    int cy = roi.y - roi.height / 2;
    window.x = cx - window_sz.width / 2;
    window.y = cy - window_sz.height / 2;
    window.width = window_sz.width;
    window.height = window_sz.height;
    return window;
}

void Fast::extractFeature(const Rect &roi, Mat image)
{
    Rect window = get_search_window_(roi, window_sz_);
    Mat z = RectTools::subwindow(image, window, BORDER_REPLICATE);
    trans.Transform(z, cnn.input_blobs()[0]);
    cnn.Forward();
}

cv::Mat createHanningMats_(int C, int H, int W) {
    cv::Mat hann1t = cv::Mat(cv::Size(H,1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,W), CV_32F, cv::Scalar(0)); 
    for (int i = 0; i < H; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (W - 1)));
    for (int i = 0; i < W; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (H - 1)));
    cv::Mat hann2d = hann2t * hann1t;
    cv::Mat hann1d = hann2d.reshape(1,1); // Procedure do deal with cv::Mat multichannel bug       
    cv::Mat hann = cv::Mat(cv::Size(H*W, C), CV_32F, cv::Scalar(0));
    for (int i = 0; i < C; i++)
        for (int j = 0; j<H*W; j++)
            hann.at<float>(i,j) = hann1d.at<float>(0,j);
    return hann;
}

cv::Mat createGaussianPeak_(int H, int W) {
    cv::Mat_<float> res(H, W);
    int syh = (H) / 2; int sxh = (W) / 2;
    float output_sigma = sqrt((float) W * H) / 2 * 0.125;// padding, output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = exp(mult * (float) (ih * ih + jh * jh));
        }
    return res;
}
