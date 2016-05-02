#!/usr/bin/python
import caffe
import os
TFP=os.path.dirname(__file__)
caffe.set_mode_cpu()

net = caffe.Net(TFP+'/kcf.prototxt', caffe.TEST)

print net.forward()
