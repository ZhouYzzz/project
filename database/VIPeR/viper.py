#!/usr/bin/python

import pandas as pd
import os
import numpy as np
from cv2 import imread, resize

_TFP = os.path.dirname(__file__)

def iread(imf,root=_TFP):
    im = imread(os.path.join(root,imf))
    assert im is not None
    im = resize(im, (128,256))
    return im.transpose((2,0,1))[np.newaxis,...]

class VIPeR():
    def __init__(self):
        self.A = pd.read_csv('a.txt',sep=' ',header=None)[0].tolist()
        self.B = pd.read_csv('b.txt',sep=' ',header=None)[0].tolist()
        assert (len(self.A) == len(self.B))
        self.load()
        pass

    def load(self):
        print '[VIPeR] loading'
        self.dA = list()
        self.dB = list()
        for iA in self.A:
            self.dA.append(iread(iA))
        for iB in self.B:
            self.dB.append(iread(iB))

        self.dA = np.vstack(self.dA)
        self.dB = np.vstack(self.dB)
        print '[VIPeR] loading finished', self.dA.shape
        (self.N,self.C,self.H,self.W) = self.dA.shape

    def get_test(self):
        randidx = np.random.randint(0,632,316)
        query = self.dA[randidx]
        gallery = self.dB[randidx]
        return query, gallery


if __name__ == "__main__":
    db = VIPeR()
    db.get_test()

