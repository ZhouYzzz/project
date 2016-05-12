#!/usr/bin/python
"""
valid if cnn tracking works
"""

import numpy as np
from numpy.fft import *
from numpy import conj, exp
from numpy.linalg import norm
import matplotlib.pyplot as plt

def main():
    target = np.ones([20,20])
    a = np.zeros([100,100])
    b = np.zeros([100,100])
    b[4:24,4:24] = target
    # b[10,10] = 1
    a[40:60,40:60] = target
    # print Gcorrelation(a,b)
    r = Lcorrelation(a,b)
    print r
    plt.pcolor(r.real)
    plt.show()
    pass

def Lcorrelation(a, b): # model = a
    c = conj(fft2(a)) * fft2(b);
    k = c
    return k

def Gcorrelation(a, b, sigma=0.6):
    c = fftshift(ifft2(conj(fft2(a)) * fft2(b)));
    d = norm(a) + norm(b) - 2 * c;
    k = exp(-1 / sigma**2 * d / d.size);
    return k

def train(x, interp):
    # xf{ii} = fft2(feat{ii});
    # kf = sum(xf{ii} .* conj(xf{ii}), 3) / numel(xf{ii});
    # alphaf{ii} = yf./ (kf+ lambda);
    k = fft2(kernel_correlation(x, x))
    alphaf = yf / k
    pass

if __name__ == '__main__':
    main()