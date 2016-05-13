#!/bin/sh
. init.sh
cp -a $ROOT/caffe/. $CAFFE/.

cp -a $ROOT/caffe/src/caffe/layers/. $FASTER_RCNN/caffe-fast-rcnn/src/caffe/layers/.
cp -a $ROOT/caffe/include/caffe/layers/. $FASTER_RCNN/caffe-fast-rcnn/include/caffe/layers/.

cd $CAFFE
make
cd $ROOT

cd $FASTER_RCNN/caffe-fast-rcnn
make
cd $ROOT
