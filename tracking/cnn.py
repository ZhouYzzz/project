#!/usr/bin/python
"""
valid if cnn tracking works
"""

import numpy as np
from numpy.fft import *
from numpy import conj, exp
from numpy.linalg import norm

def main():
    a = np.zeros([3,4,4])
    b = a.copy()
    a[0,0,0] = 1
    b[0,0,1] = 2
    b[0,0,2] = 1
    print Gcorrelation(a,b)
    print Lcorrelation(a,b)
    pass

def Lcorrelation(a, b):
    c = ifft2(sum(conj(fft2(a)) * fft2(b), 0));
    k = c
    return k

def Gcorrelation(a, b, sigma=0.6):
    c = ifft2(sum(conj(fft2(a)) * fft2(b), 0));
    d = norm(a) + norm(b) - 2 * c;
    k = exp(-1 / sigma**2 * abs(d) / d.size);
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