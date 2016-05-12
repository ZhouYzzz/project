#!/usr/bin/python
"""
valid if cnn tracking works
"""

import numpy as np
from numpy.fft import *
from numpy import conj, exp
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy import signal

def main():
    target = np.ones([10,10])
    a = np.zeros([50,50])
    b = np.zeros([50,50])
    # b[10,10] = 1
    g = signal.gaussian(50, std=3)
    prob = g[np.newaxis,:] * g[:,np.newaxis]
    yf = fft2(prob)
    h = np.hanning(50)
    hann = h[np.newaxis,:] * h[:,np.newaxis]


    a[20:30,20:30] = target
    b[4:14,4:14] = target

    kf = Lcorrelation(a,a)
    xf = fft2(a)
    alphaf = yf / (kf + 1e-4)

    kzf = Lcorrelation(b, a)
    res = (ifft2(alphaf*kzf))
     # print r[3,3]
    # print r
    plt.pcolor(res.real)
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