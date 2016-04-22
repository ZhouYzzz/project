from threading import Thread

import caffe
from utils import CHECK

class DataLayer(caffe.Layer):
    def batch_advancer(self):
        """Prepare data for `forward` in advance"""
        raise Exception(
            'You should specify your own batch_advancer')

    def init(self):
        """User specified `init` method"""
        raise Exception(
            'You should write your init function')

    def check(self, bottom, top):
        """Used for subclasses to check layer shape"""
        pass

    def setup(self, bottom, top):
        # check shape
        self.check(bottom, top)
        # worker thread
        self.worker = None
        self.buffer = None
        # excute user's `init` function
        self.init()
        # prepare buffer
        self.dispatch_worker()
        self.join_worker()

        CHECK.EQ(len(top), len(self.buffer))
        return

    def reshape(self, bottom, top):
        for i in xrange(len(top)):
            top[i].reshape(*self.buffer[i].shape)

    def forward(self, bottom, top):
        if self.worker is not None:
            self.join_worker()

        # do forward
        for i in xrange(len(top)):
            top[i].data[...] = self.buffer[i].copy()

        self.dispatch_worker()

    def backward(self, top, propagate_down, bottom):
        pass

    def dispatch_worker(self):
        '''fill the buffer'''
        CHECK.EQ(self.worker, None)
        self.worker = Thread(target=self.batch_advancer)
        self.worker.start()

    def join_worker(self):
        '''wait for worker'''
        CHECK.NEQ(self.worker, None)
        self.worker.join()
        self.worker = None
