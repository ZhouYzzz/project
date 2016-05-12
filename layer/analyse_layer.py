import caffe
import numpy as np
import pandas as pd
from util import CHECK
import matplotlib.pyplot as plt

class AnalyseLayer(caffe.Layer):
    '''Do analisis on result'''
    def setup(self, bottom, top):
        self.N = bottom[0].data.shape[0]
        pass

    def reshape(self, bottom, top):
        #top[0].reshape(self.N)
        #top[1].reshape(self.N)
        pass

    def forward(self, bottom, top):
        anchor = bottom[0].data
        pair = bottom[1].data
        label = bottom[2].data.reshape(-1)
        print anchor.shape, pair.shape

        diff = np.linalg.norm(anchor-pair,ord=2,axis=1)

        dist_sim = diff[label.astype(bool)]
        dist_diff = diff[np.logical_not(label.astype(bool))]

        top[0].reshape(*dist_sim.shape)
        top[0].data[...] = dist_sim
        top[1].reshape(*dist_diff.shape)
        top[1].data[...] = dist_diff

    def backward(self, top, propagate_down, bottom):
        pass
