#include <gflags/gflags.h>
#include <glog/logging.h>
#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cuComplex.h"
#include "cufft.h"

#include <cstdio>

#include "caffe/cukcf/recttools.hpp"

#define CUFFT_CHECK(condition) do{cufftResult result=condition;CHECK_EQ(result,CUFFT_SUCCESS)<<" "<<result;}while(0)

DEFINE_string(model, "SqueezeNet_v1.0/feat.prototxt",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "SqueezeNet_v1.0/squeezenet_v1.0.caffemodel",
    "the pretrained weights to for testing.");
DEFINE_string(vedio, "bolt",
    "vedio squence.");

using namespace cv;
using namespace std;

Size2i get_search_window(Size2i, Size2i);
cv::Mat createGaussianPeak(int H, int W);
cv::Mat createHanningMats(int C, int H, int W);  
void extract_feature(Mat im, Point2f pos, Size2i window_sz, caffe::Net<float>* cnn, caffe::DataTransformer<float>* trans);
float subPixelPeak(float left, float center, float right);

int main(int argc, char** argv)
{
    FLAGS_alsologtostderr = 1;

    caffe::GlobalInit(&argc, &argv);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(1);

	caffe::Net<float> cnn(FLAGS_model, caffe::TEST);
    cnn.CopyTrainedLayersFrom(FLAGS_weights);
	//
	caffe::TransformationParameter param;
	caffe::DataTransformer<float> trans(param, caffe::TEST);

    // ===================
    // Environment setting
    // ===================
	
    string vedio_path = "database/vot2013/"+FLAGS_vedio;
	char im_name[20]; sprintf(im_name, "/%08d.jpg", 1);
	// 207,117,29,103
    // 336,165,25,60
	Size2i target_sz(25, 60);
    Mat im = imread(vedio_path+string(im_name), CV_LOAD_IMAGE_COLOR);
    Size2i im_sz(im.cols, im.rows);
	LOG(INFO) << "target_sz:" << target_sz << "im_sz:" << im_sz;
    Size2i window_sz = get_search_window(target_sz, im_sz);
    LOG(INFO) << window_sz;
	Rect roi(336,165,25,60);
	Point2f pos(336+25.0/2, 165.0+60.0/2);

	int C, H, W, N;

	cnn.input_blobs()[0]->Reshape(1, 3, window_sz.height, window_sz.width);
	cnn.Reshape();
	// cnn.output_blobs()[0]->shape()[0]
	C = cnn.output_blobs()[0]->channels();
	H = cnn.output_blobs()[0]->height();
	W = cnn.output_blobs()[0]->width();
	N = C * H * W;
	LOG(INFO) << C << " " << H << " " << W;
	LOG(INFO) << window_sz;

	Mat prob_ = createGaussianPeak(H, W);
	Mat hann_ = createHanningMats(C, H, W);
	// ============
	// MEM SPACE
	// ============
	cufftHandle planm_;
	cufftHandle plans_;
	cublasHandle_t handle_;

	float* tf1_;
	cuComplex* tm1_;
	cuComplex* tm2_;
	cuComplex* tm3_;
	cuComplex* ts1_;
	cuComplex* ts2_;
	cuComplex* ts3_;

	cuComplex one_;
	cuComplex zero_;
	cuComplex lambda_;
	cuComplex* ones_;
	cuComplex* null_;

	cuComplex* probf;
	cuComplex* alphaf;
	cuComplex* xf;
	cuComplex* zf;
	cuComplex* feat;
	cuComplex* hann;
	float* resp;

	one_ = make_cuFloatComplex(1.0, 0.0);
	zero_ = make_cuFloatComplex(0.0, 0.0);
	lambda_ = make_cuFloatComplex(0.0001, 0.0);
	CUDA_CHECK(cudaMalloc((void**)&null_, sizeof(cuComplex)*H*W));
	CUDA_CHECK(cudaMalloc((void**)&ones_, sizeof(cuComplex)*C));
	caffe::caffe_gpu_set_C(C, one_, ones_);

	CUDA_CHECK(cudaMalloc((void**)&feat, sizeof(cuComplex)*N));
	CUDA_CHECK(cudaMalloc((void**)&hann, sizeof(cuComplex)*N));

	CUDA_CHECK(cudaMalloc((void**)&probf, sizeof(cuComplex)*H*W));
	CUDA_CHECK(cudaMalloc((void**)&xf, sizeof(cuComplex)*N));
	CUDA_CHECK(cudaMalloc((void**)&zf, sizeof(cuComplex)*N));
	CUDA_CHECK(cudaMalloc((void**)&alphaf, sizeof(cuComplex)*H*W));
	CUDA_CHECK(cudaMalloc((void**)&resp, sizeof(float)*H*W));

	CUDA_CHECK(cudaMalloc((void**)&tf1_, sizeof(float)*N));
															
	CUDA_CHECK(cudaMalloc((void**)&tm1_, sizeof(cuComplex)*N));
	CUDA_CHECK(cudaMalloc((void**)&tm2_, sizeof(cuComplex)*N));
	CUDA_CHECK(cudaMalloc((void**)&tm3_, sizeof(cuComplex)*N));

	CUDA_CHECK(cudaMalloc((void**)&ts1_, sizeof(cuComplex)*H*W));
	CUDA_CHECK(cudaMalloc((void**)&ts2_, sizeof(cuComplex)*H*W));
	CUDA_CHECK(cudaMalloc((void**)&ts3_, sizeof(cuComplex)*H*W));

	handle_ = caffe::Caffe::cublas_handle();
	int shape[2] = {H, W};
	CUFFT_CHECK(cufftPlanMany(&planm_, 2, shape, NULL, 1, H*W, NULL, 1, H*W, CUFFT_C2C, C));
	CUFFT_CHECK(cufftPlan2d(&plans_, H, W, CUFFT_C2C));

	CUDA_CHECK(cudaMemcpy(tf1_, hann_.data, sizeof(float)*N, cudaMemcpyHostToDevice));
	caffe::caffe_gpu_cpy_R2C(N, tf1_, hann);
	CUDA_CHECK(cudaMemcpy(tf1_, prob_.data, sizeof(float)*H*W, cudaMemcpyHostToDevice));
	caffe::caffe_gpu_cpy_R2C(H*W, tf1_, probf);
	CUFFT_CHECK(cufftExecC2C(plans_, probf, probf, CUFFT_FORWARD));



	// ============
	// GLOBAL VARS
	// ============

	// ============
	// RUN TRACKER
	// ============
	//
	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	
	int frame = 1;
	LOG(INFO) << "[Frame]: " << frame;
	sprintf(im_name, "/%08d.jpg", frame);
	im = imread(vedio_path+string(im_name), CV_LOAD_IMAGE_COLOR);

	extract_feature(im, pos, window_sz, &cnn, &trans);
	caffe::caffe_gpu_cpy_R2C(N, cnn.output_blobs()[0]->gpu_data(), tm1_);
	caffe::caffe_gpu_mul_C(N, tm1_, hann, feat);

	CUFFT_CHECK(cufftExecC2C(planm_, feat, xf, CUFFT_FORWARD)); // xf = fft(feat)

	caffe::caffe_gpu_mul_cjC(N, xf, xf, tm1_);
	CUBLAS_CHECK(cublasCgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_T,
				1, H*W, C, &one_, ones_, C, tm1_, H*W, &zero_, ts1_, 1)); // ts1_ = k
	// CUBLAS_CHECK(cublasCgemv(handle_, CUBLAS_OP_T, C, H*W, &one_, tm1_, C, ones_, 1, &zero_, ts1_, 1)); // ts1_ = kf
	float f = 1.0/(N*H*W);
	CUBLAS_CHECK(cublasCsscal(handle_, H*W, &f, ts1_, 1));
	
	caffe::caffe_gpu_add_scalar_C(H*W, ts1_, lambda_, ts2_); // fft(k)+lambda
	caffe::caffe_gpu_div_C(H*W, probf, ts2_, alphaf); // alphaf = probf / fft(k)+lambda

	//rectangle(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
	Point pt1(pos.x-W/2, pos.y-H/2); Point pt2(pos.x+W/2, pos.y+H/2);
	rectangle(im, pt1, pt2, Scalar(1));
	imshow( "Display window", im );
	waitKey(1);
	

	for (frame=2; frame<340; frame++) {
		// LOG(INFO) << "[Frame]: " << frame;
		sprintf(im_name, "/%08d.jpg", frame);
		im = imread(vedio_path+string(im_name), CV_LOAD_IMAGE_COLOR);

		extract_feature(im, pos, window_sz, &cnn, &trans);
		caffe::caffe_gpu_cpy_R2C(N, cnn.output_blobs()[0]->gpu_data(), tm1_);
		caffe::caffe_gpu_mul_C(N, tm1_, hann, feat);

		CUFFT_CHECK(cufftExecC2C(planm_, feat, zf, CUFFT_FORWARD)); // zf = fft(feat)

		caffe::caffe_gpu_mul_cjC(N, xf, zf, tm1_);
		CUBLAS_CHECK(cublasCgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_T,
				1, H*W, C, &one_, ones_, C, tm1_, H*W, &zero_, ts1_, 1)); // ts1_ = k
		// CUBLAS_CHECK(cublasCgemv(handle_, CUBLAS_OP_T, C, H*W, &one_, tm1_, C, ones_, 1, &zero_, ts1_, 1)); // ts1_ = kf
		CUBLAS_CHECK(cublasCsscal(handle_, H*W, &f, ts1_, 1));

		caffe::caffe_gpu_mul_C(H*W, ts1_, alphaf, ts2_);
		CUFFT_CHECK(cufftExecC2C(plans_, ts2_, ts2_, CUFFT_INVERSE));
		float fac = 1.0/H*W;
		CUBLAS_CHECK(cublasCsscal(handle_, H*W, &fac, ts2_, 1)); // scale by (1-factor)

		
		caffe::caffe_gpu_real_C(H*W, ts2_, resp);

		// TODO find location
		cv::Mat res(H, W, CV_32F);
		CUDA_CHECK(cudaMemcpy(res.data, resp, sizeof(float)*H*W, cudaMemcpyDeviceToHost));

//		LOG(INFO) << res.at<float>(0,0);
//
//		int cx = res.cols/2;
//        int cy = res.rows/2;
//
//      // rearrange the quadrants of Fourier image
//      // so that the origin is at the image center
//        Mat tmp;
//        Mat q0(res, Rect(0, 0, cx, cy));
//        Mat q1(res, Rect(cx, 0, cx, cy));
//        Mat q2(res, Rect(0, cy, cx, cy));
//        Mat q3(res, Rect(cx, cy, cx, cy));
//
//        q0.copyTo(tmp);
//        q3.copyTo(q0);
//        tmp.copyTo(q3);
//
//        q1.copyTo(tmp);
//        q2.copyTo(q1);
//        tmp.copyTo(q2);
//
//        // DO FFTSHIFT
//		LOG(INFO) << res.at<float>(0,0);


        cv::Point2i pi;
		double pv;
		cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
		float peak_value = (float) pv;

		cv::Point2f p((float)pi.x, (float)pi.y);
		if (pi.x > 0 && pi.x < W-1) {
			p.x += subPixelPeak(res.at<float>(pi.y, pi.x-1), peak_value, res.at<float>(pi.y, pi.x+1));
		}
		
		if (pi.y > 0 && pi.y < H-1) {
			p.y += subPixelPeak(res.at<float>(pi.y-1, pi.x), peak_value, res.at<float>(pi.y+1, pi.x));
		}
		p.x -= W / 2.0;
		p.y -= H / 2.0;
		pos.x += p.x;
		pos.y += p.y;
		LOG(INFO) <<"Frame:"<<frame <<";"<< p.x << "," << p.y << " pos: " << pos.x << "," << pos.y<<"peak"<<peak_value;
		// TODO find location

		extract_feature(im, pos, window_sz, &cnn, &trans);
		caffe::caffe_gpu_cpy_R2C(N, cnn.output_blobs()[0]->gpu_data(), tm1_);
		caffe::caffe_gpu_mul_C(N, tm1_, hann, feat);
	
		CUFFT_CHECK(cufftExecC2C(planm_, feat, tm1_, CUFFT_FORWARD)); // tm1_ = xf = fft(feat)

		caffe::caffe_gpu_mul_cjC(N, tm1_, tm1_, tm2_);
		CUBLAS_CHECK(cublasCgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_T,
				1, H*W, C, &one_, ones_, C, tm1_, H*W, &zero_, ts1_, 1)); // ts1_ = k
		// CUBLAS_CHECK(cublasCgemv(handle_, CUBLAS_OP_T, C, H*W, &one_, tm2_, C, ones_, 1, &zero_, ts1_, 1)); // ts1_ = kf
		CUBLAS_CHECK(cublasCsscal(handle_, H*W, &f, ts1_, 1));
	
		caffe::caffe_gpu_add_scalar_C(H*W, ts1_, lambda_, ts2_); // fft(k)+lambda
		caffe::caffe_gpu_div_C(H*W, probf, ts2_, ts1_); // alphaf = probf / fft(k)+lambda // alphaf = ts1_

		// update model
		float train_interp_factor = 0.01;
		float onemin_factor = 1-train_interp_factor;
		cuComplex factor = make_cuFloatComplex(train_interp_factor, 0);
		CUBLAS_CHECK(cublasCsscal(handle_, N, &onemin_factor, xf, 1)); // scale by (1-factor)
		CUBLAS_CHECK(cublasCsscal(handle_, H*W,&onemin_factor, alphaf, 1)); // scale by (1-factor)
		CUBLAS_CHECK(cublasCaxpy(handle_, N, &factor, tm1_, 1, xf, 1)); // tmpl += factor * convfeature
		CUBLAS_CHECK(cublasCaxpy(handle_, H*W, &factor, ts1_, 1, alphaf, 1)); // alphaf += factor * alphaf_


		Point pt1(pos.x-W/2, pos.y-H/2); Point pt2(pos.x+W/2, pos.y+H/2);
		rectangle(im, pt1, pt2, Scalar(1));
		imshow( "Display window", im );
		waitKey(1);
	}
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
        for (int j = 0; j < W; j++) {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = exp(mult * (float) (ih * ih + jh * jh));
        }
    return res;
}

cv::Mat createHanningMats(int C, int H, int W) {   
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

void extract_feature(Mat im, Point2f pos, Size2i window_sz, caffe::Net<float>* cnn, caffe::DataTransformer<float>* trans) {
	Rect extracted_roi;
	extracted_roi.x = int(pos.x) - window_sz.width/2;
	extracted_roi.y = int(pos.y) - window_sz.height/2;
	extracted_roi.width = window_sz.width;
	extracted_roi.height = window_sz.height;
	Mat z = RectTools::subwindow(im, extracted_roi, BORDER_REPLICATE);
	trans->Transform(z, cnn->input_blobs()[0]);
	cnn->Forward();
}

float subPixelPeak(float left, float center, float right) {   
	float divisor = 2 * center - right - left;
	if (divisor == 0)
		return 0;   
	return 0.5 * (right - left) / divisor;
}
