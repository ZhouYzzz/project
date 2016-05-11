import caffe
import numpy as np

REID_MODEL = '/home/zhouyz14/project/net/metric_split_cuhk03/deploy_1_batch.prototxt'
REID_WEIGHT = '/home/zhouyz14/project/net/metric_split_cuhk03/snapshots/train_val_iter_10000.caffemodel'

caffe.set_mode_gpu()
caffe.set_device(2)
net = caffe.Net(REID_MODEL, REID_WEIGHT, caffe.TEST)

def crop(im, tH, tW):
    """t: target; r: raw; o: offset."""
    (_,_,rH,rW) = im.shape
    # CHECK.GT(rH,tH); CHECK.GT(rW,tW)
    oH = np.random.randint(rH-tH+1)
    oW = np.random.randint(rW-tW+1)
    return im[:,:,oH:(oH+tH),oW:(oW+tW)]

def reid():
    net.forward()
    return

def get_feature(image):
    if (len(image.shape) == 3):
        image = image.transpose((2,0,1))[np.newaxis,...]

    image = crop(image, 240, 120)
    result = net.forward(data=image)
    return result['feat'].copy().reshape(-1)
