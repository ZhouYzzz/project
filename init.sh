#!/bin/sh
[ -e config.cfg ] \
	&& source config.cfg \
	|| { echo No \`cfg\` file found ;}

PYTHONPATH=$ROOT/utils:$ROOT/layer:$CAFFE/python:$ROOT/caffe
