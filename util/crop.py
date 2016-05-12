import numpy as np
from check import CHECK

def crop(im, tH, tW):
    """t: target; r: raw; o: offset."""
    try:
        (_,_,rH,rW) = im.shape
        CHECK.GT(rH,tH); CHECK.GT(rW,tW)
        oH = np.random.randint(rH-tH+1)
        oW = np.random.randint(rW-tW+1)
        return im[:,:,oH:(oH+tH),oW:(oW+tW)]
    except:
        (_,rH,rW) = im.shape
        CHECK.GT(rH,tH); CHECK.GT(rW,tW)
        oH = np.random.randint(rH-tH+1)
        oW = np.random.randint(rW-tW+1)
        return im[:,oH:(oH+tH),oW:(oW+tW)]
