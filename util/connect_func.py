import numpy as np
from check import CHECK

def euc(a, b):
    return -np.linalg.norm(a-b,ord=2)

def cos(a, b):
    return np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b))

def abs(a, b):
    return -np.linalg.norm(a-b,ord=1)

def con(a, b):
    pass

if __name__ == '__main__':
    a = np.array([1,2,3,4,5])
    b = np.array([2,3,4,5,6])
    CHECK.RD(euc(a, b)**2, 5)
    CHECK.RD(cos(a, b), 0.99493)
    CHECK.RD(abs(a, b), -5)