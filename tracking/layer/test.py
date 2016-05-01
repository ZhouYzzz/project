#!/usr/bin/python
import caffe
caffe.set_mode_cpu()

net = caffe.Net('kcf.prototxt', caffe.TEST)

print net.forward()
