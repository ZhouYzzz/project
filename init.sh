#!/bin/sh
[ -e config.cfg ] \
	&& source config.cfg \
	|| { echo No \`cfg\` file found ;}

export PYTHONPATH=$CAFFE/python:$ROOT/utils:$ROOT:$ROOT/database
export PATH=$PATH:$CAFFE/build/tools
