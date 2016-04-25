#!/usr/bin/python

import pandas as pd
import os
import numpy as np
from cv2 import imread, resize
from utils import CHECK, crop

_TFP = os.path.dirname(__file__)

def iread(imf,root=_TFP):
    im = imread(os.path.join(root,imf))
    assert im is not None
    im = resize(im, (128,256))
    return im.transpose((2,0,1))[np.newaxis,...]

class VIPeR():
    def __init__(self):
        """ VIPeR only have 2 images for each identity, one in a view """
        self.A = pd.read_csv(_TFP+'/a.txt',sep=' ',header=None)[0].tolist()
        self.B = pd.read_csv(_TFP+'/b.txt',sep=' ',header=None)[0].tolist()
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
        """ for test purpose """
        randidx = np.random.randint(0,632,316)
        query = self.dA[randidx]
        gallery = self.dB[randidx]
        return query, gallery

    def getA(self, idx):
        """ get im from A """
        if isinstance(idx, int):
            return self.dA[[idx]]
        else:
            return self.dA[idx]

    def getB(self, idx):
        """ get im from B """
        if isinstance(idx, int):
            return self.dB[[idx]]
        else:
            return self.dB[idx]

    def _gen_sim(self, idx, ifcrop=True, cropsize=(240,120)):
        A = self.getA(idx)
        B = self.getB(idx)
        AB = np.concatenate((A,B),axis=1)
        if (ifcrop): return crop(AB, cropsize[0], cropsize[1])
        else: return AB

    def _gen_dif(self, idx, ifcrop=True, cropsize=(240,120)):
        A = self.getA(idx)
        B = self.getB(self._rand_not_idx(idx))
        AB = np.concatenate((A,B),axis=1)
        if (ifcrop): return crop(AB, cropsize[0], cropsize[1])
        else: return AB

    def gen_valid(self, batch=128, ifcrop=True, cropsize=(240,120)):
        randidx = np.random.randint(0, self.N, batch/2)
        pair_batch = list()
        list_batch = list()
        for idx in randidx:
            # SIM
            pair_batch.append(self._gen_sim(idx, ifcrop, cropsize))
            list_batch.append(1)
            # DIF
            pair_batch.append(self._gen_dif(idx, ifcrop, cropsize))
            list_batch.append(0)
        return (np.vstack(pair_batch), np.vstack(list_batch))

    def _rand_not_idx(self, idx):
        r = np.random.randint(self.N)
        if (r == idx): return self._rand_not_idx(idx)
        else: return r


if __name__ == "__main__":
    db = VIPeR()
    db.get_test()
    print db.gen_valid().shape

