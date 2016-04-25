#!/usr/bin/python
from database import VIPeR
from utils import CHECK
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
        return 0

def test_on_VIPeR(method_instance):
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
    for i in xrange(N):
        pass

if __name__ == '__main__':
    test_on_VIPeR(default_method())
