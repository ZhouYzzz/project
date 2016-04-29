#!/bin/sh
TFP=$(dirname $0)
caffe train -solver=$TFP/solver.prototxt -gpu=2
