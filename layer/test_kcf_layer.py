import caffe
import numpy as np
from utils import CHECK
#print 'MODEUL'
class TestKCFLayer(caffe.Layer):
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
        top[0].reshape(*(2,2))
        pass

    def forward(self, bottom, top):
        #top[0].data[...] = self.test_data
        top[0].data[...] = np.ones([2,2])
        print 'Nothing with this'
        pass

    def backward(self, top, propagate_down, bottom):
        pass
