#!/bin/sh
[ -e config.cfg ] \
	&& source config.cfg \
	|| { echo No \`cfg\` file found ;}

export PYTHONPATH=$CAFFE/python:$ROOT:$FASTER_RCNN/lib
export PATH=$PATH:$CAFFE/build/tools
