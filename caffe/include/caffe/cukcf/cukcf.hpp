#include "caffe/common.hpp"
#include "cuComplex.h"
#include <opencv2/opencv.hpp>

class cuTracker {
	public:
		cuTracker();

		// Initalize tracker
		virtual void init(const cv::Rect &roi, cv::Mat image);

		// Update position based on the new frame
		virtual cv::Rect update(cv::Mat image);

	protected:
		void cnnLoad();

		void allocate_memory_space();

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
		void train(const cuComplex* x, float train_interp_factor, cuComplex* alphaf);

		// input:
		//     alphaf: model, 1*H*W
		//     z: testing sample, C*H*W
		//     x: training sample, C*H*W
		// output:
		//     resp: response map, 1*H*W
		void detect(const cuComplex* alphaf, const cuComplex* z, const cuComplex* x, cuComplex* resp);

		// input:
		//     a: C*H*W
		//     b: C*H*W
		// output:
		//	   dst: 1*H*W
		void linearCorrelation(const cuComplex* a, const cuComplex* b, cuComplex* dst);

	private:
};


// help functions for complex
namespace caffe {
// a + b
void caffe_gpu_add_C(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst);
// a - b
void caffe_gpu_sub_C(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst);
// a .* b
void caffe_gpu_mul_C(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst);
// conj(a) .* b
void caffe_gpu_mul_cjC(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst);
// a + c
void caffe_gpu_add_scalar_C(const int N, const cuComplex* a, cuComplex alpha, 
		cuComplex* dst);
// a / b
void caffe_gpu_div_C(const int N, const cuComplex* a, const cuComplex* b,
		cuComplex* dst);
}
