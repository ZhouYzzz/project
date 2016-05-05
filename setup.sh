#!/bin/sh
. init.sh
cp -a $ROOT/caffe/. $CAFFE/.
cd $CAFFE
make -j48
cd $ROOT
