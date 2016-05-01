#!/bin/sh
. init.sh
cp -a $ROOT/caffe/. $CAFFE/.
cd $CAFFE
make
cd $ROOT
