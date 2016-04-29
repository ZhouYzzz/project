'''
function alphaf = train(x, y, sigma, lambda) 
    k = kernel_correlation(x, x, sigma); 
    alphaf = fft2(y) ./ (fft2(k) + lambda);
end

function responses = detect(alphaf, x, z, sigma) 
    k = kernel_correlation(z, x, sigma);
    responses = real(ifft2(alphaf .* fft2(k)));
end

function k = kernel_correlation(x1, x2, sigma)
    c = ifft2(sum(conj(fft2(x1)) .* fft2(x2), 3)); 
    d = x1(:)'*x1(:) + x2(:)'*x2(:) - 2 * c;
    k = exp(-1 / sigma^2 * abs(d) / numel(d));
end
'''

from numpy.fft import *
import numpy as np

def alphaf(x, y, sigma, lam):
    k = kernel_correlation(x, x, sigma)
    alphaf = fft2(y) / (fft2(k) + lam)
    return alphaf

def kernel_correlation(x1, x2, sigma):
    fft2(x1)
    pass

def W(xf, yf, lam=0):
    return (yf*xf.conj())/(xf*xf.conj()+lam)

def RM(Wf, zf):
    return ifft(Wf*zf.conj())

if __name__ == '__main__':
    x1 = np.array([
        [1,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]])
    x2 = np.array([
        [0,0,0,0],
        [0,0,1,0],
        [0,0,0,0],
        [0,0,0,0]])
    yf = np.hanning(4)
    print(yf)
    exit()
    xf1 = fft(x1)
    xf2 = fft(x2)
    # print yf
    # print xf1
    Wf = W(xf1, yf)
    rm = RM(Wf, xf2)
    print rm.real









