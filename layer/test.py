"""test.py"""
# from check import CHECK
from threading import Thread

import os
def path():
    print os.path.dirname(__file__), __file__

class a():
    def __init__(self):
        self.worker = None
        return

    def forward(self):
        if self.worker is not None:
            self.join_worker()

        # do forward

        self.dispatch_worker()
        return

    def batch_advancer(self):
        pass

    def dispatch_worker(self):
        '''fill the buffer'''
        print(1)
        # CHECK.EQ(self.worker, None)
        self.worker = Thread(target=self.batch_advancer)
        self.worker.start()
        return

    def join_worker(self):
        '''wait for worker'''
        print(2)
        # CHECK.NEQ(self.worker, None)
        self.worker.join()
        self.worker = None
        return

class A():
    def __init__(self):
        self.worker = None
        self.init_data_source()
        self.batch_advancer()

    def batch_advancer(self):
        print 1
        pass

    def init_data_source(self):
        pass

    def dispatch_worker(self):
        '''fill the buffer'''
        # CHECK.EQ(self.worker, None)
        self.worker = Thread(target=self.batch_advancer)
        self.worker.start()
        return

    def join_worker(self):
        '''wait for worker'''
        # CHECK.NEQ(self.worker, None)
        self.worker.join()
        self.worker = None
        return

class B(A):
    def batch_advancer(self):
        print self.data_source

    def init_data_source(self):
        self.data_source = None

