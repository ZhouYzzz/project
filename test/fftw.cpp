#include <fftw3.h>
#include <ctime>
#include <iostream>

//#define C 128
#define H 256
#define W 256

int main(int argc, char** argv) {
	fftw_complex *in, *out;
	fftw_plan p;
	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*H*W);
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*H*W);
//	p = fftw_plan_dft_2d(H, W, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	int rank = 2;
	int n[2] = {H,W};
	//fftw_plan_many_dft(rank, n, C, in, n, 1, H*W, out, n, 1, H*W, FFTW_FORWARD, FFTW_ESTIMATE);
	p = fftw_plan_dft_2d(H, W, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	std::clock_t start;
	double duration;

	start = std::clock();
	
	for (int i=0; i<64*100; i++)
		fftw_execute(p);

	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout << duration/100.0 << '\n';

	fftw_destroy_plan(p);
	return 0;
}
