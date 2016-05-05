#include "caffe/caffe.hpp"
#include "cuComplex.h"
#include "cufft.h"
#include <opencv2/opencv.hpp>

using caffe::Net;
using std::string;

class cuTracker {
	public:
		cuTracker(string model, string weights);
		~cuTracker() {}

		// Initalize tracker
		virtual void init(const cv::Rect &roi, cv::Mat image);

		// Update position based on the new frame
		virtual cv::Rect update(cv::Mat image);

		Net<float> cnn;
	protected:
		string cnnInitCheck(string model, string weights);
		void cnnLoad(string weights);

		void allocate_memory_space();

		void init_cuda_handle();

		void init_constants();

		// input:
		//     image: vedio frame
		// output:
		//     dst: feature extracted, C*H*W
		void getFeature(const cv::Mat image, cuComplex* dst);

		// input:
		//	   x: training sample, C*H*W 
		//     factor: learning rate
		// output:
		//     alphaf: trained model, 1*H*W
		void train(const cuComplex* x);

		// input:
		//     alphaf: model, 1*H*W
		//     z: testing sample, C*H*W
		//     x: training sample, C*H*W
		// output:
		//     resp: response map, 1*H*W
		void detect(const cuComplex* z);

		// input:
		//     a: C*H*W
		//     b: C*H*W
		// output:
		//	   dst: 1*H*W
		void linearCorrelation(const cuComplex* a, const cuComplex* b, cuComplex* dst);


		// constants
		cuComplex lambda;
		float interp_factor;
		cuComplex interp_factor_C;
		float onemin_factor; // 1 - interp_factor

		// operation shape
		int C, H, W;
		int N; // N = C * H * W

	private:
		// cuda plans
		cufftHandle planm_; // multichannel cufft plan
		cufftHandle plans_; // singlechannel cufft plan
		cublasHandle_t handle_; // cublas handle

		// constants
		cuComplex one_;
		cuComplex* ones_; // length: 1*H*W, for sum operation
		cuComplex zero_;
		cuComplex* null_;

		float* resp_; // detect response

		// cuda protected mem space
		// multi-channel
		cuComplex* tmpl_; // target template

		// single-channel
		cuComplex* alphaf_; // target model in F domain
		cuComplex* k_;
		cuComplex* prob_; // score

		// cuda tmp mem space, m for multi-channel, s for single-channel
		cuComplex* tm1_;
		cuComplex* tm2_;
		cuComplex* tm3_;

		cuComplex* ts1_;
		cuComplex* ts2_;
		cuComplex* ts3_;
};
