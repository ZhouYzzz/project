import caffe
import numpy as np

from fast_rcnn.test import im_detect
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms

DETECT_MODEL = '/home/zhouyz14/py-faster-rcnn/'
DETECT_WEIGHT = '/home/zhouyz14/py-faster-rcnn/'

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

CONF_THRESH = 0.8
NMS_THRESH = 0.3

# NET
caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(DETECT_MODEL, DETECT_WEIGHT, caffe::TEST)

# FUNCTIONS
def high_prob(dets, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    return dets[inds,:]

def person_detection(im):
    scores, boxes = im_detect(net, im)

    # cls_ind = 14 + 1 # 14 -- 'person'

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # skip background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
    return high_prob(dets, CONF_THRESH)