def method(im):
    """ im -> feat """
    feat = im
    return feat

import pandas as pd
import numpy as np
import os
from skimage.io import imread

_TFP = os.path.dirname(__file__)
A_TXT = 'a.txt'
B_TXT = 'b.txt'

def iread(iname, dir=_TFP):
    im = imread(os.path.join(dir,iname))
    return im.transpose((2,0,1))

class VIPeR():
    def __init__(self):
        self.A = pd.read_csv(A_TXT, header=None)[0].tolist()
        self.B = pd.read_csv(B_TXT, header=None)[0].tolist()
        self.C, self.H, self.W = iread(self.A[0]).shape
        print self.C, self.H, self.W

    def load(self):
        # print A, B
        pass

    def preprocess(self):
        pass

    def test(self):
        testidx = np.random.choice(632, 316, False)
        print testidx
        pass

import os
#print len(os.listdir('cam_a'))
#print len(os.listdir('cam_b'))

"""Cross database"""

a = VIPeR()
a.load()
a.test()