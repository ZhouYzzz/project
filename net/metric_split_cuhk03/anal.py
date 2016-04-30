#!/usr/bin/python
import caffe
import numpy as np
import matplotlib.pyplot as plt

caffe.set_mode_gpu()
caffe.set_device(2)

dist_sim = np.array([],dtype=np.float32)
dist_diff = np.array([],dtype=np.float32)

net = caffe.Net('anal_test.prototxt','snapshots/train_val_iter_3000.caffemodel',caffe.TEST)

N = 100

for i in xrange(N):
    result = net.forward()
    dist_sim = np.append(dist_sim, result['dist_sim'].copy())
    dist_diff = np.append(dist_diff, result['dist_diff'].copy())


plt.subplot(2, 1, 1)
plt.hist(dist_sim, bins=100, normed=True)
plt.xlim(0,3)
plt.title('distance distribution of SIMILAR pairs')

plt.subplot(2, 1, 2)
plt.hist(dist_diff, bins=100, normed=True)
plt.xlim(0,3)
plt.title('distance distribution of DIFFERENT pairs')

plt.show()
