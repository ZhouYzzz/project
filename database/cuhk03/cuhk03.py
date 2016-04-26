#!/usr/bin/python

import numpy as np
import os
from utils import CHECK, crop
from cv2 import imread
from random import getrandbits

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
    # print dir, iname
    im = imread(os.path.join(dir,iname))
    assert im is not None
    return im.transpose((2,0,1))[np.newaxis,...]

class CUHK03():
    '''INIT functions'''
    def __init__(self, NI=3898): # 400 classes
        self.N, self.inames, self.labels, self.NC = cuhk03_reader(_SRC, NI)
        # print iread(self.inames[0]).shape
        _, self.C, self.H, self.W = iread(self.inames[0]).shape
        pass

    def load(self):
        # allocate mem space
        print '[CUHK03]: Loading', self.N, 'images,', self.NC, 'Class.'
        # init cls idx
        self.clsidx = self.init_cls2idx()
        # self.data = np.zeros([self.N,self.C,self.H,self.W],np.uint8)
        self.data = list()
        for idx in xrange(self.N):
            # self.data[idx] = iread(self.inames[idx])
            self.data.append(iread(self.inames[idx]))

        print '[CUHK03]: Finished'

        # print len(self.data), self.data[0].shape

    '''GET functions'''
    def get(self, idx):
        return self.data[idx], self.labels[idx]

    def getd(self, idx):
        assert isinstance(idx, int)
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
        for i in xrange(self.NC):
            idx = np.where(self.labels == i)[0].tolist()
            CHECK.GT(len(idx),1)
            clsidx.append(idx)

        return clsidx

    def cls2idx(self, cls):
        return self.clsidx[cls]

    def choice(self, cls, n):
        '''gen `n` idx of cls'''
        return np.random.choice(self.cls2idx(cls), n)

    def rand_not_cls(self, cls):
        CHECK.GT(cls,0); CHECK.LT(cls,self.NC)
        r = np.random.randint(*[0,self.NC])
        if (r==cls):
            return self.rand_not_cls(cls)
        else:
            return r

    def rand_not_cls_test(self, cls):
        exit() # under construction
        CHECK.GT(cls,300); CHECK.LT(cls,400)
        r = np.random.randint(*[300,400])
        if (r==cls):
            return self.rand_not_cls(cls)
        else:
            return r

    '''GENERATE functions'''
    def gen_batch(self, batch):
        randcls = np.random.choice(np.arange(self.NC),batch)
        data = list()
        lst = list()
        for cls in randcls:
            (d, l) = self.gen_pair(cls)
            data.append(d)
            lst.append(l)
        return (data, lst)

    def gen_pair(self, cls, ifcrop=True, cropsize=(240,120)):
        '''genpair'''
        if bool(getrandbits(1)):
            return (self.gen_sim(cls, ifcrop, cropsize),1)
        else:
            return (self.gen_dif(cls, ifcrop, cropsize),0)

    def gen_pair_test(self, cls):
        pass

    def gen_sim(self, cls, ifcrop=True, cropsize=(240,120)):
        [iA, iB] = self.choice(cls, 2)
        A = self.getd(iA)
        B = self.getd(iB)
        if (ifcrop):
            A = crop(A,cropsize[0],cropsize[1])
            B = crop(B,cropsize[0],cropsize[1])
        # print A.shape, B.shape
        AB = np.concatenate((A,B),axis=1)
        # if (ifcrop): return crop(AB,cropsize[0],cropsize[1])
        return AB

    def gen_dif(self, cls, ifcrop=True, cropsize=(240,120)):
        [iA] = self.choice(cls, 1)
        [iB] = self.choice(self.rand_not_cls(cls), 1)
        A = self.getd(iA)
        B = self.getd(iB)
        # print A.shape, B.shape
        if (ifcrop):
            A = crop(A,cropsize[0],cropsize[1])
            B = crop(B,cropsize[0],cropsize[1])
        # print A.shape, B.shape
        AB = np.concatenate((A,B),axis=1)
        # if (ifcrop): return crop(AB,cropsize[0],cropsize[1])
        return AB

    def gen_sim_test(self, cls):
        pass

    def gen_dif_test(self, cls):
        pass

    def gen_triplet(self, cls):
        pass

if __name__ == "__main__":
    c = CUHK03()
    c.load()
    from time import time
    t = time()
    N = 200
    c.gen_batch(N)
    print 'Loading',N,'pairs took',time() - t
