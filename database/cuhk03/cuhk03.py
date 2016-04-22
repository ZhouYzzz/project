import numpy as np
import os
from utils import CHECK, crop
from cv2 import imread

_TFP = os.path.dirname(__file__)
_SRC = os.path.join(_TFP,'labeled.txt')

CLSINFO = \
"""
CLS - IDX
=== - ===
100 - 960
200 - 1935
300 - 2912
400 - 3898
500 - 4878
"""

def cuhk03_reader(src, NI):
    import pandas as pd
    df = pd.read_csv(src, sep=' ', header=None)[:NI]
    num = df.shape[0]
    inames = df[0].values
    labels = df[1].values
    num_cls = labels[-1] + 1
    return num, inames, labels, num_cls

def iread(iname, dir=_TFP):
    im = imread(os.path.join(dir,iname))
    return im.transpose((2,0,1))

class CUHK03():
    '''INIT functions'''
    def __init__(self, NI=3898): # 400 classes
        self.N, self.inames, self.labels, self.NC = cuhk03_reader(_SRC)
        self.C, self.H, self.W = iread(self.inames[0]).shape
        pass

    def load(self):
        # allocate mem space
        print '[CUHK03]: Loading', self.N, 'images,', self.NC, 'Class.'
        self.clsidx = self.init_cls2idx()
        self.data = np.zeros([self.N,self.C,self.H,self.W],np.uint8)
        for idx in xrange(NI):
            self.data[idx] = iread(self.inames[idx])
    
    '''GET functions'''
    def get(self, idx):
        return self.data[idx], self.labels[idx]

    def getd(self, idx):
        if isinstance(idx, int):
            return self.data[[idx]]

        return self.data[idx]
    
    def getl(self, idx):
        return self.labels[idx]

    def get4d(self, idx):
        self.getd(idx)

    def get3d(self, idx):
        assert isinstance(idx, int)
        return self.data[idx]

    '''INDEX functions'''
    def init_cls2idx(self):
        clsidx = list()
        for i in xrange(400):
            idx = np.where(self.labels == i)[0].tolist()
            CHECK.GT(len(idx),1)
            clsidx.append(idx)

        return clsidx

    def cls2idx(self, cls):
        return self.clsidx[cls]

    def rand_not_cls(self, cls):
        CHECK.GT(cls,0); CHECK.LT(cls,300)
        r = np.random.randint(*[0,300])
        if (r==cls):
            return self.rand_not_cls(cls)
        else:
            return r

    def rand_not_cls_test(self, cls):
        CHECK.GT(cls,300); CHECK.LT(cls,400)
        r = np.random.randint(*[300,400])
        if (r==cls):
            return self.rand_not_cls(cls)
        else:
            return r

    '''GENERATE functions'''
    def gen_batch(self, batch):
        randcls = np.random.choice(np.arange(0,300),batch)
        data = list()
        lst = list()
        for cls in randcls:
            (d, l) = self.gen_pair(cls)
        pass

    def gen_pair(self, cls):
        pass

    def gen_pair_test(self, cls):
        pass

    def gen_sim(self, cls):
        pass

    def gen_dif(self, cls):
        pass

    def gen_sim_test(self, cls):
        pass

    def gen_dif_test(self, cls):
        pass

    def gen_triplet(self, cls):
        pass

if __name__ == "__main__":
    c = CUHK03()
    c.load()
