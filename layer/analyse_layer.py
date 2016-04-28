import caffe
import numpy as np
import pandas as pd
from utils import CHECK

class AnalyseLayer(caffe.Layer):
    '''Do analisis on result'''
    def setup(self, bottom, top):
        self.N = bottom[0].data.shape
        pass

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        anchor = bottom[0].data
        pair = bottom[1].data
        label = bottom[2].data

        diff = np.linalg.norm(anchor-pair,ord=2,axis=1)

        dist_sim = list()
        dist_diff= list()
        for i in xrange(self.N):
            if label[i]: # sim pair
                dist_sim.append(diff[i])
            else: # diff pair
                dist_diff.append(diff[i])

    def backward(self, top, propagate_down, bottom):
        pass