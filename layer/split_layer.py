from util import CHECK
import caffe

class SplitLayer(caffe.Layer):
    '''Split the data into several parts'''
    def setup(self, bottom, top):
        CHECK.EQ(len(bottom), 1)
        CHECK.EQ(len(top), 3) # we spilt 1 im to 3 parts
        self.N, self.C, self.H, self.W = bottom[0].shape
        # S -- Slice: `H`eight / 2   
        self.S = self.H / 2
        pass

    def reshape(self, bottom, top):
        for i in xrange(3):
            top[i].reshape(*(self.N, self.C, self.S, self.W))

    def forward(self, bottom, top):
        S = self.S
        top[0].data[...] = bottom[0].data[:,:,:S,:]
        top[1].data[...] = bottom[0].data[:,:,S/2:-S/2,:]
        top[2].data[...] = bottom[0].data[:,:,S:,:]

    def backward(self, top, propagate_down, bottom):
        pass
