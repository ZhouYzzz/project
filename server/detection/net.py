import sys
# sys.path.remove('/home/zhouyz14/project')
sys.path.remove('/home/zhouyz14/caffe/python')
sys.path.append('/home/zhouyz14/py-faster-rcnn/caffe-fast-rcnn/python')
import caffe
import numpy as np

from fast_rcnn.test import im_detect
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
# ~/py-faster-rcnn/models/coco/VGG16/faster_rcnn_end2end$ ls ../../../../data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel
DETECT_MODEL = '/home/zhouyz14/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt'
DETECT_WEIGHT = '/home/zhouyz14/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel'

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

CONF_THRESH = 0.6
NMS_THRESH = 0.3

cfg.TEST.HAS_RPN = True
# NET
caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(DETECT_MODEL, DETECT_WEIGHT, caffe.TEST)

# after loading, remove path change
# sys.path.append('/home/zhouyz14/project')
sys.path.append('/home/zhouyz14/caffe/python')
sys.path.remove('/home/zhouyz14/py-faster-rcnn/caffe-fast-rcnn/python')

# FUNCTIONS
def high_prob(dets, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    return dets[inds,:]

def person_detection(im):
    scores, boxes = im_detect(net, im)

    cls_ind = 14 + 1 # 14 -- 'person'

    #for cls_ind, cls in enumerate(CLASSES[1:]):
    # for cls_ind, cls in [(14, 'person')]:
    # cls_ind += 1 # skip background
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    # print cls_scores
    dets = np.hstack((cls_boxes,
                    cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    return high_prob(dets, CONF_THRESH)
