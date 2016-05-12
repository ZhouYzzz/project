import caffe
import numpy as np
from utils import CHECK
from numpy.fft import *
#print 'MODEUL'

C=3
H=5
W=5

class RefLayer(caffe.Layer):
    def setup(self, bottom, top):
        #print 'HAHAAH'
        #CHECK.EQ(len(top), 1)
        #self.test_data = np.arange(128).reshape(*(8,4,2,2))
        #self.test_data = np.ones([1,1])
        #print self.test_data
        pass

    def reshape(self, bottom, top):
        #print 'NONONO'
        #top[0].reshape(*self.test_data.shape)
        top[0].reshape(1,C,H,W)
        pass

    def forward(self, bottom, top):
        #top[0].data[...] = self.test_data
        a = np.zeros([C,H,W])
        a[0,1,1] = 1
        a[0,1,2] = 1
        a[1,1,1] = 1
        a[2,2,2] = 1
        #a[2,2,3] = 1
        print a
        af =  fft2(a)
        c =  af.conj()*af
        print af.real, c.real
        s = np.sum(c, axis=0)
        print s.real

        #b = np.zeros([1,C,H,W])
        #b[0,:,1:2,2:3] = b[0,:,1:2,2:3] + 1
        #print b
        top[0].data[0,:,:,:] = a
        # print 'Nothing with this'
        pass

    def backward(self, top, propagate_down, bottom):
        pass
