import caffe
from data_layer import DataLayer
from database import CUHK03
import numpy as np

class Metric3PartsDataLayer(DataLayer):
    '''devide the person into 3 parts: up, mid, btm'''
    def check(self, bottom, top):
        CHECK.EQ(len(top), 2)
        pass

"""
Well, we can add a split layer instead.
"""