#!/usr/bin/python
from database import VIPeR
from utils import CHECK, crop
import numpy as np

class default_method():
    def __init__(self):
        pass

    def __call__(self, data):
        return data.reshape(data.shape[0],-1)

class deep_method():
    def __init__(self, proto, model):
        import caffe
        self.net = caffe.Net(proto, model, caffe.TEST)

    def __call__(self, data):
        num = data.shape[0]
        shape = self.net.blobs['data'].data.shape
        crop_data = np.zeros([num,shape[1],shape[2],shape[3]],np.uint8)
        for i in xrange(num):
            crop_data[i] = crop(data[i],shape[2],shape[3])

        self.net.blobs['data'].reshape(*crop_data.shape)
        return self.net.forward(data=crop_data)['feat'].reshape(num,-1).copy()

def L2_connect(f, f_gallery):
    dist = f_gallery - f
    return -np.linalg.norm(dist, ord=2, axis=1)

def test_on_VIPeR(method_instance, connect_func=L2_connect):
    db = VIPeR()
    query, gallery = db.get_test()
    N = query.shape[0]

    print '[TEST] generate feature'
    query_feat = method_instance(query)
    gallery_feat = method_instance(gallery)
    #print gallery_feat, query_feat
    CHECK.EQ(N, query_feat.shape[0])
    CHECK.EQ(N, gallery_feat.shape[0])
    CHECK.EQ(2, len(query_feat.shape))
    CHECK.EQ(2, len(gallery_feat.shape))

    print '[TEST] begin ranking'
    CMC = np.zeros(N)
    for i in xrange(N):
        qf = query_feat[i]
        # connect function: great means similar
        similarity = connect_func(qf, gallery_feat)
        rank = np.where((-similarity).argsort() == i)[0][0]
        CMC[rank:] += 1

    CMC = CMC/N
    return CMC

if __name__ == '__main__':
    test_on_VIPeR(default_method())
