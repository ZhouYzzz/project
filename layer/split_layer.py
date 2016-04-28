import caffe

class SplitLayer(caffe.Layer):
    '''Split the data into several parts'''
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        pass
    
    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass