#!/usr/bin/python
import sys
try:
    sys.argv[1]
except:
    print 'ERROR: No target prototxt specified.'
    exit()

import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(sys.argv[1], caffe.TEST)
