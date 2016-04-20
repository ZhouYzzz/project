import numpy as np
import os
from skimage.io import imread

_TFP = os.path.dirname(__file__)
_SRC = os.path.join(_TFP,'labeled.txt')
_DIR = os.path.join(_TFP,'labeled')

def cuhk03_reader(src):
    import pandas as pd
    df = pd.read_csv(src, sep=' ', header=None)
    num = df.shape[0]
    inames = df[0].values
    lables = df[1].values
    return num, inames, labels

def iread(iname, dir=_DIR):
    im = imread(os.path.join(dir,iname))
    return im.transpose((2,0,1))

class CUHK03():
    def __init__(self):
        self.N, self.inames, self.labels = cuhk03_reader(_SRC)
        self.C, self.H, self.W = iread(self.inames[0]).shape
        pass

    def load(self):
        # allocate mem space
        self.data = np.zeros([self.N,self.C,self.H,self.W],np.uint8)
        for idx in xrange(self.N):
            self.data[idx] = iread(self.inames[idx])

    def get(self, idx):
        return self.data[idx], self.labels[idx]