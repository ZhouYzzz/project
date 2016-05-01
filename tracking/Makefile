ROOT:=1

CUDA:=/usr/local/cuda

all:
	nvcc -I$(CUDA)/include -lcufft cukcf.cu -o cukcf
